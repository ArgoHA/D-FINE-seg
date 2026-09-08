#pragma once
// NVENC (HEVC, CUDA device-pointer input, stream-ordered via nvEncSetIOCudaStreams) with a ring of
// `depth` NV12 input frames + bitstream buffers, muxed to MP4 through libavformat.
#include <cuda.h>

#include <string>
#include <vector>

#include "kernels.h"

extern "C" {
#include <libavformat/avformat.h>
}
#include <ffnvcodec/nvEncodeAPI.h>

struct NvencFunctions;

struct NvEncOpts {
  std::string preset = "P4";  // P1..P7
  bool low_latency = false;   // tuning: high_quality (default) or low_latency
  int extra_delay = 3;        // ring slack beyond B-frames + lookahead (NvEncoder sample default)
  int bitrate = 0;            // bits/s average (VBR); 0 = preset default
};

class NvEncoder {
 public:
  NvEncoder(const std::string& out_path, int w, int h, double fps, CUcontext ctx, CUstream stream,
            NvencFunctions* nv, const NvEncOpts& opts);
  ~NvEncoder();
  int depth() const { return depth_; }
  Nv12View slot(int i) const { return slots_[i]; }
  // Encode the frame in slot (frame_idx % depth). Blocks only when the ring is full: the output
  // of frame (frame_idx - depth + 1) is drained here, which is what frees that slot for reuse.
  void submit(long frame_idx);
  void finish();  // EOS, drain, write trailer
  long packets = 0;
  int frame_interval_p = 0, lookahead = 0;

 private:
  void retrieve(long frame_idx);
  void write_packet(void* data, int size, long long pts_ticks, bool key);

  NV_ENCODE_API_FUNCTION_LIST fn_{};
  void* enc_ = nullptr;
  CUstream stream_;
  int w_, h_, depth_ = 0, reorder_ = 0;
  long submitted_ = 0, retrieved_ = 0;
  std::vector<Nv12View> slots_;
  std::vector<CUdeviceptr> mem_;
  std::vector<NV_ENC_REGISTERED_PTR> reg_;
  std::vector<NV_ENC_INPUT_PTR> mapped_;
  std::vector<NV_ENC_OUTPUT_PTR> bs_;
  AVFormatContext* ofmt_ = nullptr;
  AVStream* st_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVRational tick_{1, 1};
  bool finished_ = false;
};
