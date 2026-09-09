// Core-only throughput: how fast can this board run the engine, with no video pipeline in the
// way. Frames are synthetic and already on the GPU, so this measures preprocess + engine +
// postprocess and nothing else - the number to size a Jetson/embedded target with.
//
//   dfine_bench model.engine [--width 1920] [--height 1080] [--iters 400] [--slots 1,2,4,8]
//
// "blocking" is dfine_infer (one sync per frame, what most consumers write first); "async" is
// dfine_infer_async on N slots from N threads, which is the same work with the syncs removed.
#include <dfine.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s model.engine [--width W] [--height H] [--iters N] [--slots 1,2,4,8]\n",
                 argv[0]);
    return 2;
  }
  int W = 1920, H = 1080, iters = 400;
  std::vector<int> slot_counts;
  try {
    for (int i = 2; i < argc; ++i) {
      auto need = [&]() -> const char* {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return argv[++i];
      };
      std::string k = argv[i];
      if (k == "--width") W = std::atoi(need());
      else if (k == "--height") H = std::atoi(need());
      else if (k == "--iters") iters = std::atoi(need());
      else if (k == "--slots") {
        slot_counts.clear();
        for (char* t = std::strtok(const_cast<char*>(need()), ","); t; t = std::strtok(nullptr, ","))
          slot_counts.push_back(std::atoi(t));
      } else throw std::runtime_error("unknown arg " + k);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  if (slot_counts.empty()) slot_counts = {1, 2, 4, 8};
  if (W <= 0 || H <= 0 || iters <= 0) {
    std::fprintf(stderr, "--width/--height/--iters must be positive\n");
    return 2;
  }

  DfineModel* m = dfine_open(argv[1], 0);
  if (!m) {
    std::fprintf(stderr, "open failed: %s\n", dfine_last_error());
    return 1;
  }
  const DfineInfo* in = dfine_info(m);
  const char* tn = in->task == DFINE_SEM_SEG ? "sem_seg"
                   : in->task == DFINE_SEGMENT ? "segment" : "detect";

  uint8_t* dev_img = nullptr;  // a frame already on the GPU, like a camera/ISP hand-off
  if (cudaMalloc(&dev_img, (size_t)W * H * 3) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    return 1;
  }
  cudaMemset(dev_img, 128, (size_t)W * H * 3);
  std::printf("%s engine (input %dx%d), %dx%d source frame, %d iters\n", tn, in->in_w, in->in_h, W,
              H, iters);

  {  // blocking: one slot, sync every frame
    DfineSlot* s = dfine_slot_create(m);
    if (!s) {
      std::fprintf(stderr, "slot failed: %s\n", dfine_last_error());
      return 1;
    }
    for (int i = 0; i < 20; ++i) dfine_infer(m, s, dev_img, W, H, 0, DFINE_RGB8, 1, nullptr);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) dfine_infer(m, s, dev_img, W, H, 0, DFINE_RGB8, 1, nullptr);
    double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  blocking, 1 slot   %7.3f ms/frame  %7.1f fps\n", ms / iters,
                1000.0 * iters / ms);
    dfine_slot_destroy(s);
  }

  for (int n : slot_counts) {  // async: n slots, n threads, no per-frame sync
    if (n <= 0) continue;
    std::vector<DfineSlot*> slots(n);
    std::vector<cudaStream_t> st(n);
    for (int i = 0; i < n; ++i) {
      slots[i] = dfine_slot_create(m);  // captures a CUDA graph: done before the threads start
      if (!slots[i]) {
        std::fprintf(stderr, "slot failed: %s\n", dfine_last_error());
        return 1;
      }
      cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking);
    }
    auto run = [&](int i, int reps) {
      for (int r = 0; r < reps; ++r)
        dfine_infer_async(m, slots[i], dev_img, W, H, 0, DFINE_RGB8, 1, st[i]);
      cudaStreamSynchronize(st[i]);
    };
    for (int i = 0; i < n; ++i) run(i, 10);
    const int per = iters / n > 0 ? iters / n : 1;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int i = 0; i < n; ++i) th.emplace_back(run, i, per);
    for (auto& t : th) t.join();
    double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  async, %-2d slot%s    %7.3f ms/frame  %7.1f fps\n", n, n > 1 ? "s" : " ",
                ms / (per * n), 1000.0 * (per * n) / ms);
    for (int i = 0; i < n; ++i) {
      dfine_slot_destroy(slots[i]);
      cudaStreamDestroy(st[i]);
    }
  }
  cudaFree(dev_img);
  dfine_close(m);
  return 0;
}
