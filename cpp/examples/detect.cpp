// Minimal consumer of the core API: load an engine, run one image, print the detections.
//   dfine_example model.engine frame.ppm [conf]
// PPM (P6) only, so the example needs no image library; produce one with
//   ffmpeg -i frame.jpg -pix_fmt rgb24 frame.ppm
#include <dfine.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static std::vector<uint8_t> read_ppm(const char* path, int& w, int& h) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + std::string(path));
  std::string magic;
  int maxv = 0;
  f >> magic >> w >> h >> maxv;
  if (magic != "P6" || maxv != 255) throw std::runtime_error("not an 8-bit P6 PPM");
  f.get();  // single whitespace before the pixel data
  std::vector<uint8_t> px((size_t)w * h * 3);
  f.read((char*)px.data(), px.size());
  if (!f) throw std::runtime_error("short read");
  return px;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.engine frame.ppm [conf]\n", argv[0]);
    return 2;
  }
  int w = 0, h = 0;
  std::vector<uint8_t> img;
  try {
    img = read_ppm(argv[2], w, h);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

  DfineModel* m = dfine_open(argv[1], 0);
  if (!m) {
    std::fprintf(stderr, "open failed: %s\n", dfine_last_error());
    return 1;
  }
  const DfineInfo* info = dfine_info(m);
  const char* task = info->task == DFINE_SEM_SEG ? "sem_seg"
                     : info->task == DFINE_SEGMENT ? "segment" : "detect";
  std::printf("engine: %s, input %dx%d, top-K %d | image %dx%d\n", task, info->in_w, info->in_h,
              info->max_det, w, h);
  if (argc > 3) dfine_set_conf(m, (float)atof(argv[3]));

  DfineSlot* s = dfine_slot_create(m);
  if (!s) {
    std::fprintf(stderr, "slot failed: %s\n", dfine_last_error());
    dfine_close(m);
    return 1;
  }

  if (info->task == DFINE_SEM_SEG) {
    // The label map stays on the device; copy it only because we want to print it.
    if (dfine_infer(m, s, img.data(), w, h, 0, DFINE_RGB8, 0, nullptr) != DFINE_OK) {
      std::fprintf(stderr, "infer failed: %s\n", dfine_last_error());
      return 1;
    }
    std::vector<int32_t> map((size_t)info->sem_h * info->sem_w);
    cudaMemcpy(map.data(), dfine_sem(s), map.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<int> hist(256, 0);
    for (int32_t v : map)
      if (v >= 0 && v < 256) ++hist[v];
    std::printf("label map %dx%d, classes present:", info->sem_h, info->sem_w);
    for (int c = 0; c < 256; ++c)
      if (hist[c]) std::printf(" %d(%.1f%%)", c, 100.0 * hist[c] / map.size());
    std::printf("\n");
  } else {
    DfineDets d;
    if (dfine_infer(m, s, img.data(), w, h, 0, DFINE_RGB8, 0, &d) != DFINE_OK) {
      std::fprintf(stderr, "infer failed: %s\n", dfine_last_error());
      return 1;
    }
    std::printf("%d detections (boxes in image pixels)\n", d.count);
    for (int i = 0; i < d.count; ++i)
      std::printf("  cls %-3d %.3f  [%.1f %.1f %.1f %.1f]\n", d.labels[i], d.scores[i],
                  d.boxes[i * 4], d.boxes[i * 4 + 1], d.boxes[i * 4 + 2], d.boxes[i * 4 + 3]);
    // Masks stayed on the GPU. Ask for them only if you need them:
    if (info->task == DFINE_SEGMENT && d.count > 0) {
      uint8_t* dev = nullptr;
      cudaMalloc(&dev, (size_t)d.count * h * w);
      dfine_upsample_masks(m, s, dev, d.count, h, w, 0.5f, 0);
      cudaStreamSynchronize(0);
      std::vector<uint8_t> mask((size_t)h * w);
      cudaMemcpy(mask.data(), dev, mask.size(), cudaMemcpyDeviceToHost);
      size_t on = 0;
      for (uint8_t v : mask) on += v;
      std::printf("  instance 0 mask: %zu/%zu px set at %dx%d\n", on, mask.size(), w, h);
      cudaFree(dev);
    }
  }

  dfine_slot_destroy(s);
  dfine_close(m);
  return 0;
}
