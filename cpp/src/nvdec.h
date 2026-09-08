#pragma once
// libavformat demux -> cuvid parser -> NVDEC. Every displayed picture is copied out of NVDEC's
// surface inside the parser callback (the parser recycles decode surfaces as soon as the display
// callback returns) into an NV12 ring the consumer reads asynchronously.
#include <cuda.h>
#include <cuda_runtime.h>

#include <deque>
#include <string>
#include <vector>

#include "kernels.h"

extern "C" {
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
}
#include <ffnvcodec/dynlink_cuviddec.h>
#include <ffnvcodec/dynlink_nvcuvid.h>

struct CuvidFunctions;

class NvDecoder {
 public:
  NvDecoder(const std::string& path, CUcontext ctx, CuvidFunctions* cv, CUstream copy_stream,
            int display_delay, int ring = 8, bool no_copy = false);
  ~NvDecoder();
  // Next frame in display order; `slot` stays valid until release(). False at end of stream.
  bool next(Nv12View& out, int& slot);
  // Record `consumer`'s progress; the ring slot is reused only after that work completed.
  void release(int slot, cudaStream_t consumer);
  int width = 0, height = 0;  // display size from the bitstream (what the frame ring holds)
  double fps = 0;
  bool bt709 = true;  // what PyNvVideoCodec assumes for untagged streams
  long frames = 0;

 private:
  static int seq_cb(void* u, CUVIDEOFORMAT* f);
  static int dec_cb(void* u, CUVIDPICPARAMS* p);
  static int disp_cb(void* u, CUVIDPARSERDISPINFO* d);
  bool pump();  // feed one packet (or EOS) into the parser; false when nothing more to feed
  void parse(CUVIDSOURCEDATAPACKET* p);
  void destroy();  // also runs when the constructor throws

  CUcontext ctx_;
  CuvidFunctions* cv_;
  CUstream stream_;
  AVFormatContext* fmt_ = nullptr;
  AVBSFContext* bsf_ = nullptr;
  AVPacket* pkt_ = nullptr;
  int vidx_ = -1;
  CUvideoparser parser_ = nullptr;
  CUvideodecoder dec_ = nullptr;
  CUvideoctxlock lock_ = nullptr;
  std::deque<int> pending_;
  std::string err_;
  bool eof_ = false, eos_sent_ = false;
  int surf_h_ = 0, disp_w_ = 0, disp_h_ = 0, disp_x_ = 0, disp_y_ = 0;
  int ring_;
  bool no_copy_;  // benchmark mode: map/unmap without copying out (NVDEC-only ceiling)
  long released_ = 0;
  std::vector<Nv12View> slots_;
  std::vector<cudaEvent_t> ev_;
  std::vector<bool> ev_set_;
};
