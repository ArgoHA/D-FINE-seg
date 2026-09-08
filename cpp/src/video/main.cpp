// GPU-resident pipeline: NVDEC -> TensorRT (batch 1, CUDA graph) -> NV12 overlay (boxes, instance
// masks or a dense label map, by engine type) -> NVENC, one worker thread per clip, everything
// stream-ordered (the only host syncs are the per-frame decoder unmap and NVENC's blocking output
// lock once the ring is full).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "engine.h"
#include "kernels_video.h"
#include "nvdec.h"
#include "nvenc.h"
#include <ffnvcodec/dynlink_loader.h>

namespace fs = std::filesystem;

// Blend strengths, mirroring the Annotator in scripts/dfine_gpu_end2end.py (not CLI-settable).
constexpr float kBoxAlpha = 0.6f, kMaskBodyAlpha = 0.45f, kMaskEdgeAlpha = 0.70f, kSemAlpha = 0.5f;

struct Args {
  std::string engine = "./model.engine";
  std::string videos = "./test_videos";
  std::string out;  // default: <videos>_annotated
  std::string dump;  // optional dir for per-frame detections (parity checks)
  std::string dump_input;  // optional file: raw fp32 [3,H,W] engine input of frame dump_input_i
  std::string dump_map;  // optional prefix: mask owner map (uint16 [mh,mw]) / sem_seg map (int32) per frame
  long dump_input_i = 0;
  std::vector<long> dump_map_i;
  int n_classes = 0;  // palette size; 0 = 8, or 19 for sem_seg (graphs carry no class count)
  std::vector<int> labels_to_use;  // empty = all
  float conf = 0.5f, nms_iou = 0.7f, mask_scale = 0.5f;
  int workers = 8, max_dim = 1920, display_delay = 4, gpu = 0;
  std::string stage = "full";  // prefix of the pipeline to run; see --help
  NvEncOpts enc;
};

static std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t p = 0;
  while (p <= s.size()) {
    size_t q = s.find(sep, p);
    if (q == std::string::npos) q = s.size();
    if (q > p) out.push_back(s.substr(p, q - p));
    p = q + 1;
  }
  return out;
}

static Args parse(int argc, char** argv) {
  Args a;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--engine") a.engine = need(i);
    else if (k == "--videos") a.videos = need(i);
    else if (k == "--out") a.out = need(i);
    else if (k == "--dump") a.dump = need(i);
    else if (k == "--dump-input") { a.dump_input = need(i); a.dump_input_i = std::stol(need(i)); }
    else if (k == "--dump-map") { a.dump_map = need(i); for (auto& t : split(need(i), ',')) a.dump_map_i.push_back(std::stol(t)); }
    else if (k == "--n-classes") a.n_classes = std::stoi(need(i));
    else if (k == "--labels") {
      for (auto& t : split(need(i), ',')) {  // the class filter is a 64-bit mask
        int l = std::stoi(t);
        if (l < 0 || l >= 64) throw std::runtime_error("--labels id out of range (0-63): " + t);
        a.labels_to_use.push_back(l);
      }
    }
    else if (k == "--conf") a.conf = std::stof(need(i));
    else if (k == "--nms") a.nms_iou = std::stof(need(i));
    else if (k == "--mask-scale") a.mask_scale = std::stof(need(i));
    else if (k == "--workers") a.workers = std::stoi(need(i));
    else if (k == "--max-dim") a.max_dim = std::stoi(need(i));
    else if (k == "--display-delay") a.display_delay = std::stoi(need(i));
    else if (k == "--preset") a.enc.preset = need(i);
    else if (k == "--low-latency") a.enc.low_latency = true;
    else if (k == "--enc-delay") a.enc.extra_delay = std::stoi(need(i));
    else if (k == "--bitrate") a.enc.bitrate = std::stoi(need(i));
    else if (k == "--gpu") a.gpu = std::stoi(need(i));
    else if (k == "--stage") a.stage = need(i);
    else if (k == "-h" || k == "--help") {
      std::printf(
          "dfine_e2e [--engine E] [--videos DIR] [--out DIR] [--n-classes N] [--labels 0,2]\n"
          "          [--conf 0.5] [--nms 0.7] [--mask-scale 0.5] [--workers 8] [--max-dim 1920]\n"
          "          [--display-delay 4] [--preset P4] [--low-latency] [--enc-delay 3] [--bitrate BPS]\n"
          "          [--gpu 0] [--stage parse|decode|infer|draw|full|bench]\n"
          "          [--dump DIR] [--dump-input FILE FRAME] [--dump-map PREFIX FRAME,FRAME,..]\n"
          "\nThe engine must be exported with keep_ratio=False: the preprocess kernel squishes the\n"
          "frame to the engine input, it has no letterbox path (like TRTModel.gpu_run).\n"
          "\nThe task follows the engine: labels/boxes/scores = detect, + masks = instance segmentation\n"
          "(masks computed at --mask-scale of the output size, like the Python pipeline), a single\n"
          "int32 map = semantic segmentation. --n-classes sizes the colour palette (default 8, or 19\n"
          "for sem_seg); --labels keeps only those class ids (0-63).\n"
          "\n--stage runs a prefix of the pipeline, for locating the bottleneck:\n"
          "  parse  NVDEC only (map/unmap, no copy-out)      draw  + overlay\n"
          "  decode + copy into the frame ring               full  + NVENC and MP4 muxing\n"
          "  infer  + preprocess, engine and postprocess     bench isolated infer path, no video\n"
          "");
      std::exit(0);
    } else throw std::runtime_error("unknown arg " + k);
  }
  if (a.out.empty()) a.out = a.videos + "_annotated";
  if (a.stage != "full" && a.stage != "draw" && a.stage != "infer" && a.stage != "decode" &&
      a.stage != "parse" && a.stage != "bench")
    throw std::runtime_error("--stage must be parse, decode, infer, draw, full or bench");
  if (a.display_delay < 0 || a.enc.extra_delay < 0)
    throw std::runtime_error("--display-delay and --enc-delay must be >= 0");
  const bool infers = a.stage == "full" || a.stage == "draw" || a.stage == "infer";
  if (!infers && (!a.dump.empty() || !a.dump_input.empty() || !a.dump_map.empty()))
    throw std::runtime_error("--dump/--dump-input/--dump-map need --stage infer, draw or full");
  return a;
}

// compute_out_size from the Python script: long edge <= max_dim, even dims, half-even rounding.
static void out_size(int w, int h, int max_dim, int& ow, int& oh) {
  ow = w; oh = h;
  if (max_dim > 0 && std::max(w, h) > max_dim) {
    double s = (double)max_dim / std::max(w, h);
    ow = std::max(2, (int)std::nearbyint(w * s / 2) * 2);
    oh = std::max(2, (int)std::nearbyint(h * s / 2) * 2);
  }
  ow &= ~1; oh &= ~1;
}

// Visualizer.generate_colors: hues on a violet->red arc via OpenCV's 8-bit HSV->RGB.
static void class_rgb(int i, int n, int& r, int& g, int& b) {
  int denom = std::max(n - 1, 1);
  int hue = (int)(135LL * (n - 1 - i) / denom);
  float h = hue * (6.f / 180.f), s = 230.f / 255.f, v = 200.f / 255.f;
  static const int sector_data[6][3] = {{1, 3, 0}, {1, 0, 2}, {3, 0, 1}, {0, 2, 1}, {0, 1, 3}, {2, 1, 0}};
  int sector = (int)std::floor(h);
  h -= sector;
  float tab[4] = {v, v * (1.f - s), v * (1.f - s * h), v * (1.f - s * (1.f - h))};
  auto q = [](float x) { return (int)std::lround(x * 255.f); };
  b = q(tab[sector_data[sector][0]]);
  g = q(tab[sector_data[sector][1]]);
  r = q(tab[sector_data[sector][2]]);
}

// rgb_to_nv12 constants from the Python script (BT.709 limited range, truncating cast).
static void rgb_to_yuv(int r, int g, int b, uint8_t* yuv) {
  auto c = [](float x) { return (uint8_t)std::min(std::max(x, 0.f), 255.f); };
  yuv[0] = c(0.1826f * r + 0.6142f * g + 0.0620f * b + 16.f);
  yuv[1] = c(-0.1006f * r - 0.3386f * g + 0.4392f * b + 128.f);
  yuv[2] = c(0.4392f * r - 0.3989f * g - 0.0403f * b + 128.f);
}

struct Shared {
  Args args;
  CUcontext ctx = nullptr;
  CuvidFunctions* cv = nullptr;
  NvencFunctions* nv = nullptr;
  TrtEngine* engine = nullptr;
  uint8_t* palette = nullptr;  // device [n_classes][3] Y,U,V
  int n_classes = 0;
  uint64_t class_mask;
  int thick;
  std::vector<fs::path> clips;
  std::atomic<size_t> next_clip{0};
  std::atomic<long> frames{0};
  std::mutex log_mu;
};

struct Worker {
  cudaStream_t main = nullptr, dec = nullptr;
  TrtContext* tctx = nullptr;
  Dets* dets = nullptr;
  uint16_t* owner = nullptr;  // segment: [mh, mw] instance-id map
  size_t owner_cap = 0;
};

// Per-clip geometry: output frame, engine->frame box scale, and the mask grid (a fraction of the
// output size, MASK_SCALE in the Python script; Python's round() is half-even like nearbyint).
struct Geo {
  int ow, oh, mh, mw;
  float sx, sy, bsx, bsy;
};
static Geo geometry(const Args& a, const TrtEngine& e, int src_w, int src_h) {
  Geo g;
  out_size(src_w, src_h, a.max_dim, g.ow, g.oh);
  g.mh = std::max(1, (int)std::nearbyint(g.oh * a.mask_scale));
  g.mw = std::max(1, (int)std::nearbyint(g.ow * a.mask_scale));
  g.sx = (float)g.ow / e.in_w; g.sy = (float)g.oh / e.in_h;
  g.bsx = (float)g.mw / e.in_w; g.bsy = (float)g.mh / e.in_h;
  return g;
}

static void ensure_owner(Worker& w, const Geo& g) {
  size_t need = (size_t)g.mh * g.mw;
  if (need <= w.owner_cap) return;
  if (w.owner) cudaFree(w.owner);
  w.owner = nullptr;
  w.owner_cap = 0;
  CK(cudaMalloc(&w.owner, need * sizeof(uint16_t)));
  w.owner_cap = need;
}

// Per-clip device/pinned scratch: freed on every exit path, so a clip that throws (bad codec,
// ring overflow) does not leak GPU or pinned memory in a worker that moves on to the next one.
struct Scratch {
  cudaStream_t s;
  std::vector<Nv12View> frames;  // painted double buffer when NVENC is off
  std::vector<cudaEvent_t> ev;   // --dump: detections copied out
  std::vector<Dets*> hdets;
  explicit Scratch(cudaStream_t stream) : s(stream) {}
  ~Scratch() {
    cudaStreamSynchronize(s);  // buffers may still be referenced by queued work
    for (auto& v : frames) cudaFree(v.y);
    for (auto e : ev) cudaEventDestroy(e);
    for (auto* p : hdets) cudaFreeHost(p);
  }
};

// Everything after the engine, per task; `outv` is the frame being painted.
static void annotate(Shared& sh, Worker& w, const Geo& g, const Nv12View& outv, bool draw) {
  const Args& a = sh.args;
  const TrtEngine& e = *sh.engine;
  if (e.sem_seg) {
    if (draw)
      draw_sem_seg(outv, w.tctx->sem, e.sem_h, e.sem_w, sh.palette, sh.n_classes, sh.class_mask, kSemAlpha, w.main);
    return;
  }
  // sx=sy=1: the overlay path keeps boxes in engine-input space and scales them per frame
  // (g.sx, g.sy) into the output frame, which is a different size from the source.
  postprocess(w.tctx->labels, e.labels_i64, w.tctx->boxes, w.tctx->scores, e.k, a.conf, sh.class_mask,
              a.nms_iou, 1.f, 1.f, w.dets, w.main);
  if (e.has_masks) {
    mask_owners(w.dets, w.tctx->masks, e.mask_h, e.mask_w, g.mh, g.mw, g.bsx, g.bsy, w.owner, w.main);
    if (draw)
      draw_masks(outv, w.dets, w.owner, g.mh, g.mw, sh.palette, sh.n_classes, kMaskBodyAlpha,
                 kMaskEdgeAlpha, w.main);
  }
  if (draw) draw_boxes(outv, w.dets, g.sx, g.sy, sh.palette, sh.n_classes, sh.thick, kBoxAlpha, w.main);
}

// Isolated timing of the inference path on one resident frame: no decode, no encode, so the
// per-frame cost of preprocess / engine / postprocess is visible instead of hidden behind NVDEC.
static void bench_infer(Shared& sh, Worker& w, const fs::path& clip) {
  const Args& a = sh.args;
  const TrtEngine& e = *sh.engine;
  NvDecoder dec(clip.string(), sh.ctx, sh.cv, w.dec, a.display_delay);
  Nv12View src;
  int slot;
  if (!dec.next(src, slot)) throw std::runtime_error("no frames");
  Geo g = geometry(a, e, dec.width, dec.height);
  ensure_owner(w, g);
  Scratch sc(w.main);
  uint8_t* outp = nullptr;
  CK(cudaMalloc(&outp, (size_t)g.ow * g.oh * 3 / 2));
  Nv12View outv{outp, outp + (size_t)g.ow * g.oh, g.ow, g.ow, g.oh};
  sc.frames.push_back(outv);
  const int N = 2000;
  auto time_it = [&](const char* name, int steps) {
    for (int warm = 0; warm < 2; ++warm) {
      int iters = warm ? N : 20;
      CK(cudaStreamSynchronize(w.main));
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        nv12_to_input(src, w.tctx->input, e.in_h, e.in_w, dec.bt709, w.main);
        if (steps > 1) w.tctx->run(w.main);
        if (steps > 2) {
          if (steps > 3) nv12_resize(src, outv, w.main);
          annotate(sh, w, g, outv, steps > 3);
        }
      }
      CK(cudaStreamSynchronize(w.main));
      if (!warm) continue;
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      LOG("  %-34s %6.3f ms/frame  (%7.1f fps)", name, ms / iters, 1000.0 * iters / ms);
    }
  };
  LOG("isolated infer path (%s), 1 stream, %dx%d source:", e.task(), src.w, src.h);
  time_it("preprocess (NV12->engine fp32 CHW)", 1);
  time_it("+ engine (CUDA graph replay)", 2);
  time_it(e.sem_seg ? "+ (no postprocess)" : e.has_masks ? "+ postprocess (NMS + mask owners)" : "+ postprocess (thresh+NMS)", 3);
  char buf[64];
  std::snprintf(buf, sizeof buf, "+ resize to %dx%d and draw", g.ow, g.oh);
  time_it(buf, 4);
}

static long process_clip(Shared& sh, Worker& w, const fs::path& clip) {
  const Args& a = sh.args;
  const TrtEngine& e = *sh.engine;
  const bool do_copy = a.stage != "parse";
  const bool do_infer = do_copy && a.stage != "decode", do_draw = do_infer && a.stage != "infer";
  const bool do_encode = a.stage == "full";
  NvDecoder dec(clip.string(), sh.ctx, sh.cv, w.dec, a.display_delay, 8, !do_copy);  // copies on w.dec
  Geo g = geometry(a, e, dec.width, dec.height);
  if (e.has_masks) ensure_owner(w, g);
  fs::path out = fs::path(a.out) / clip.parent_path().filename() / (clip.stem().string() + "_annotated.mp4");
  std::unique_ptr<NvEncoder> enc;
  if (do_encode) {
    fs::create_directories(out.parent_path());
    enc.reset(new NvEncoder(out.string(), g.ow, g.oh, dec.fps, sh.ctx, w.main, sh.nv, a.enc));
  }
  // Without the encoder there is no NVENC input ring, so paint into a private double buffer.
  Scratch sc(w.main);
  const int R = do_encode ? enc->depth() : 2;
  if (!do_encode)
    for (int j = 0; j < R; ++j) {
      uint8_t* p = nullptr;
      CK(cudaMalloc(&p, (size_t)g.ow * g.oh * 3 / 2));
      sc.frames.push_back(Nv12View{p, p + (size_t)g.ow * g.oh, g.ow, g.ow, g.oh});
    }

  std::ofstream dump;
  if (!a.dump.empty()) {
    fs::create_directories(a.dump);
    dump.open((fs::path(a.dump) / (clip.parent_path().filename().string() + "_" + clip.stem().string() + ".txt")).string());
    for (int j = 0; j < R; ++j) {
      cudaEvent_t ev = nullptr;
      Dets* hd = nullptr;
      CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
      sc.ev.push_back(ev);
      CK(cudaMallocHost((void**)&hd, sizeof(Dets)));
      sc.hdets.push_back(hd);
    }
  }
  auto write_dump = [&](long i) {  // boxes in output-frame pixels, like the Python dump
    const Dets* d = sc.hdets[i % R];
    dump << i << ' ' << d->count;
    for (int j = 0; j < d->count; ++j)
      dump << ' ' << d->boxes[j * 4] * g.sx << ' ' << d->boxes[j * 4 + 1] * g.sy << ' ' << d->boxes[j * 4 + 2] * g.sx
           << ' ' << d->boxes[j * 4 + 3] * g.sy << ' ' << d->labels[j] << ' ' << d->scores[j];
    dump << '\n';
  };
  auto dump_dev = [&](const std::string& path, const void* src, size_t bytes) {
    std::vector<char> h(bytes);
    CK(cudaMemcpyAsync(h.data(), src, bytes, cudaMemcpyDeviceToHost, w.main));
    CK(cudaStreamSynchronize(w.main));
    std::ofstream(path, std::ios::binary).write(h.data(), bytes);
  };

  long n = 0;
  for (long i = 0;; ++i) {
    Nv12View src;
    int slot;
    if (!dec.next(src, slot)) break;
    const int s = (int)(i % R);
    if (dump.is_open() && i >= R) {
      CK(cudaEventSynchronize(sc.ev[s]));
      write_dump(i - R);
    }
    if (!do_copy) { ++n; continue; }
    Nv12View outv = do_encode ? enc->slot(s) : sc.frames[s];
    if (do_infer) nv12_to_input(src, w.tctx->input, e.in_h, e.in_w, dec.bt709, w.main);
    nv12_resize(src, outv, w.main);
    if (i == a.dump_input_i && !a.dump_input.empty())
      dump_dev(a.dump_input, w.tctx->input, (size_t)3 * e.in_h * e.in_w * 4);
    dec.release(slot, w.main);  // decoder may overwrite `src` once these two kernels are done
    if (do_infer) {
      w.tctx->run(w.main);
      annotate(sh, w, g, outv, do_draw);
      if (std::find(a.dump_map_i.begin(), a.dump_map_i.end(), i) != a.dump_map_i.end()) {
        // Stem included like --dump: without it two clips in one directory race to write
        // the same file and the dump silently reports whichever worker finished last.
        std::string f = a.dump_map + "_" + clip.parent_path().filename().string() + "_" +
                        clip.stem().string() + "_" + std::to_string(i) + ".bin";
        if (e.sem_seg) dump_dev(f, w.tctx->sem, (size_t)e.sem_h * e.sem_w * 4);
        else if (e.has_masks) dump_dev(f, w.owner, (size_t)g.mh * g.mw * 2);
      }
    }
    if (dump.is_open()) {
      CK(cudaMemcpyAsync(sc.hdets[s], w.dets, sizeof(Dets), cudaMemcpyDeviceToHost, w.main));
      CK(cudaEventRecord(sc.ev[s], w.main));
    }
    if (do_encode) enc->submit(i);  // NVENC waits on w.main (nvEncSetIOCudaStreams)
    ++n;
  }
  if (do_encode) enc->finish();
  CK(cudaStreamSynchronize(w.main));
  if (dump.is_open())
    for (long i = std::max(0L, n - R); i < n; ++i) {
      CK(cudaEventSynchronize(sc.ev[i % R]));
      write_dump(i);
    }
  {
    std::lock_guard<std::mutex> lk(sh.log_mu);
    if (!do_encode) {
      LOG("Processed %s (%ld frames, stage=%s)", clip.c_str(), n, a.stage.c_str());
    } else {
      LOG("Saved annotated video: %s (%ld frames in, %ld packets out)", out.c_str(), n, enc->packets);
      if (enc->packets != n) LOG("  WARNING: frame count mismatch");
    }
  }
  return n;
}

static void worker_main(Shared& sh, Worker& w) {
  CU(cuCtxSetCurrent(sh.ctx));
  long local = 0;
  for (;;) {
    size_t i = sh.next_clip.fetch_add(1);
    if (i >= sh.clips.size()) break;
    try {
      if (sh.args.stage == "bench") { bench_infer(sh, w, sh.clips[i]); break; }
      local += process_clip(sh, w, sh.clips[i]);
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lk(sh.log_mu);
      LOG("Clip failed: %s: %s", sh.clips[i].c_str(), e.what());
    }
  }
  sh.frames += local;
}

int main(int argc, char** argv) {
  Shared sh;
  try {
    sh.args = parse(argc, argv);
    const Args& a = sh.args;
    CK(cudaSetDevice(a.gpu));
    CK(cudaFree(nullptr));  // create the primary context
    CU(cuInit(0));
    CUdevice dev;
    CU(cuDeviceGet(&dev, a.gpu));
    CU(cuDevicePrimaryCtxRetain(&sh.ctx, dev));
    CU(cuCtxSetCurrent(sh.ctx));
    if (cuvid_load_functions(&sh.cv, nullptr) < 0) throw std::runtime_error("libnvcuvid not found");
    if (nvenc_load_functions(&sh.nv, nullptr) < 0) throw std::runtime_error("libnvidia-encode not found");

    // Clips: <videos>/<cam>/*.mp4 (sorted), plus any *.mp4 directly under <videos>.
    std::vector<fs::path> dirs;
    for (auto& d : fs::directory_iterator(a.videos)) if (d.is_directory()) dirs.push_back(d.path());
    std::sort(dirs.begin(), dirs.end());
    dirs.insert(dirs.begin(), fs::path(a.videos));
    for (auto& d : dirs) {
      std::vector<fs::path> v;
      for (auto& f : fs::directory_iterator(d)) {
        std::string ext = f.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (f.is_regular_file() && ext == ".mp4") v.push_back(f.path());
      }
      std::sort(v.begin(), v.end());
      sh.clips.insert(sh.clips.end(), v.begin(), v.end());
    }
    if (sh.clips.empty()) throw std::runtime_error("no .mp4 clips under " + a.videos);
    int n_workers = a.stage == "bench" ? 1 : std::max(1, std::min(a.workers, (int)sh.clips.size()));
    LOG("%zu clips; %d workers", sh.clips.size(), n_workers);

    TrtEngine engine(a.engine);
    sh.engine = &engine;
    if (engine.k > kMaxDet) throw std::runtime_error("engine top-K exceeds kMaxDet");
    if (engine.sem_seg && !a.dump.empty())
      throw std::runtime_error("--dump has no detections on a sem_seg engine (use --dump-map)");
    if (!engine.sem_seg && !engine.has_masks && !a.dump_map.empty())
      throw std::runtime_error("--dump-map has no map on a detect engine (use --dump)");
    if (engine.sem_seg)
      LOG("engine: %s, input %dx%d, label map %dx%d", engine.task(), engine.in_w, engine.in_h, engine.sem_w, engine.sem_h);
    else
      LOG("engine: %s, input %dx%d, top-K %d, labels %s%s", engine.task(), engine.in_w, engine.in_h, engine.k,
          engine.labels_i64 ? "int64" : "int32", engine.has_masks ? ", masks" : "");

    // Graph exports carry no class count: --n-classes, else cityscapes' 19 for sem_seg and 8.
    sh.n_classes = a.n_classes > 0 ? a.n_classes : engine.sem_seg ? 19 : 8;
    int nc = sh.n_classes;
    std::vector<uint8_t> pal(nc * 3);
    for (int i = 0; i < nc; ++i) {
      int r, g, b;
      class_rgb(i, nc, r, g, b);
      rgb_to_yuv(r, g, b, &pal[i * 3]);
    }
    CK(cudaMalloc(&sh.palette, pal.size()));
    CK(cudaMemcpy(sh.palette, pal.data(), pal.size(), cudaMemcpyHostToDevice));
    sh.class_mask = 0;  // 0 = every class; parse() rejected ids outside [0, 63]
    for (int l : a.labels_to_use) sh.class_mask |= 1ULL << l;
    sh.thick = std::max(1, (a.max_dim > 0 ? a.max_dim : 1920) / 400);

    // Per-worker streams + TRT contexts (graphs captured here, before any other thread runs CUDA).
    std::vector<Worker> workers(n_workers);
    std::vector<std::unique_ptr<TrtContext>> ctxs;
    for (auto& w : workers) {
      CK(cudaStreamCreateWithFlags(&w.main, cudaStreamNonBlocking));
      CK(cudaStreamCreateWithFlags(&w.dec, cudaStreamNonBlocking));
      ctxs.emplace_back(new TrtContext(engine, w.main));
      w.tctx = ctxs.back().get();
      CK(cudaMalloc(&w.dets, sizeof(Dets)));
      CK(cudaMemset(w.dets, 0, sizeof(Dets)));
    }
    CK(cudaDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (auto& w : workers) threads.emplace_back(worker_main, std::ref(sh), std::ref(w));
    for (auto& t : threads) t.join();
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    long total = sh.frames.load();
    LOG("Processed %ld frames from %zu clips in %.1fs (%.1f frames/s aggregate)", total, sh.clips.size(),
        el, el > 0 ? total / el : 0.0);

    for (auto& w : workers) {
      cudaFree(w.dets);
      if (w.owner) cudaFree(w.owner);
      cudaStreamDestroy(w.main);
      cudaStreamDestroy(w.dec);
    }
    ctxs.clear();
    cudaFree(sh.palette);
  } catch (const std::exception& e) {
    LOG("fatal: %s", e.what());
    return 1;
  }
  return 0;
}
