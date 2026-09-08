#include "nvdec.h"

#include <stdexcept>

#include "common.h"
#include <ffnvcodec/dynlink_loader.h>

static cudaVideoCodec av_to_cuvid(AVCodecID id) {
  switch (id) {
    case AV_CODEC_ID_H264: return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC: return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP9: return cudaVideoCodec_VP9;
    case AV_CODEC_ID_AV1: return cudaVideoCodec_AV1;
    case AV_CODEC_ID_MPEG4: return cudaVideoCodec_MPEG4;
    default: throw std::runtime_error("unsupported codec for NVDEC");
  }
}

NvDecoder::NvDecoder(const std::string& path, CUcontext ctx, CuvidFunctions* cv, CUstream s,
                     int display_delay, int ring, bool no_copy)
    : ctx_(ctx), cv_(cv), stream_(s), ring_(ring), no_copy_(no_copy) {
  if (avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr) < 0)
    throw std::runtime_error("avformat_open_input failed: " + path);
  if (avformat_find_stream_info(fmt_, nullptr) < 0) throw std::runtime_error("no stream info");
  vidx_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (vidx_ < 0) throw std::runtime_error("no video stream: " + path);
  AVStream* st = fmt_->streams[vidx_];
  AVCodecParameters* par = st->codecpar;
  width = par->width;
  height = par->height;
  fps = av_q2d(st->r_frame_rate.num ? st->r_frame_rate : st->avg_frame_rate);  // nominal, like ffprobe

  const char* bsf_name = par->codec_id == AV_CODEC_ID_H264   ? "h264_mp4toannexb"
                         : par->codec_id == AV_CODEC_ID_HEVC ? "hevc_mp4toannexb"
                                                             : "null";
  const AVBitStreamFilter* filt = av_bsf_get_by_name(bsf_name);
  if (!filt || av_bsf_alloc(filt, &bsf_) < 0) throw std::runtime_error("bsf alloc failed");
  avcodec_parameters_copy(bsf_->par_in, par);
  bsf_->time_base_in = st->time_base;
  if (av_bsf_init(bsf_) < 0) throw std::runtime_error("bsf init failed");
  pkt_ = av_packet_alloc();

  CU(cuCtxSetCurrent(ctx_));
  CU(cv_->cuvidCtxLockCreate(&lock_, ctx_));
  CUVIDPARSERPARAMS pp{};
  pp.CodecType = av_to_cuvid(par->codec_id);
  pp.ulMaxNumDecodeSurfaces = 1;
  pp.ulMaxDisplayDelay = display_delay;
  pp.pUserData = this;
  pp.pfnSequenceCallback = seq_cb;
  pp.pfnDecodePicture = dec_cb;
  pp.pfnDisplayPicture = disp_cb;
  CU(cv_->cuvidCreateVideoParser(&parser_, &pp));
}

NvDecoder::~NvDecoder() {
  if (parser_) cv_->cuvidDestroyVideoParser(parser_);
  if (dec_) cv_->cuvidDestroyDecoder(dec_);
  if (lock_) cv_->cuvidCtxLockDestroy(lock_);
  if (pkt_) av_packet_free(&pkt_);
  if (bsf_) av_bsf_free(&bsf_);
  if (fmt_) avformat_close_input(&fmt_);
  for (auto& v : slots_) cudaFree(v.y);
  for (auto e : ev_) cudaEventDestroy(e);
}

int NvDecoder::seq_cb(void* u, CUVIDEOFORMAT* f) {
  auto* self = (NvDecoder*)u;
  if (self->dec_) return f->min_num_decode_surfaces;  // no reconfigure support; keep going
  if (f->chroma_format != cudaVideoChromaFormat_420 || f->bit_depth_luma_minus8 != 0) {
    self->err_ = "only 8-bit 4:2:0 is supported";
    return 0;
  }
  CUVIDDECODECREATEINFO ci{};
  ci.CodecType = f->codec;
  ci.ChromaFormat = f->chroma_format;
  ci.OutputFormat = cudaVideoSurfaceFormat_NV12;
  ci.bitDepthMinus8 = f->bit_depth_luma_minus8;
  ci.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
  ci.ulNumOutputSurfaces = 2;
  ci.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  ci.ulNumDecodeSurfaces = f->min_num_decode_surfaces;
  ci.vidLock = self->lock_;
  ci.ulWidth = f->coded_width;
  ci.ulHeight = f->coded_height;
  ci.ulMaxWidth = f->coded_width;
  ci.ulMaxHeight = f->coded_height;
  ci.ulTargetWidth = f->coded_width;  // full coded surface; the visible rect is read out below
  ci.ulTargetHeight = f->coded_height;
  self->surf_h_ = f->coded_height;
  self->disp_x_ = f->display_area.left;
  self->disp_y_ = f->display_area.top;
  self->disp_w_ = (f->display_area.right - f->display_area.left) & ~1;
  self->disp_h_ = (f->display_area.bottom - f->display_area.top) & ~1;
  unsigned m = f->video_signal_description.matrix_coefficients;
  self->bt709 = !(m == 5 || m == 6);  // 470bg / smpte170m -> BT.601, everything else BT.709
  CUresult r = self->cv_->cuvidCreateDecoder(&self->dec_, &ci);
  if (r != CUDA_SUCCESS) {
    self->err_ = "cuvidCreateDecoder failed: " + std::to_string((int)r);
    return 0;
  }
  try {
    for (int i = 0; i < self->ring_; ++i) {
      uint8_t* p = nullptr;
      size_t w = self->disp_w_, h = self->disp_h_;
      CK(cudaMalloc(&p, w * h * 3 / 2));
      self->slots_.push_back(Nv12View{p, p + w * h, (int)w, (int)w, (int)h});
      cudaEvent_t e;
      CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      self->ev_.push_back(e);
      self->ev_set_.push_back(false);
    }
  } catch (const std::exception& e) {
    self->err_ = e.what();
    return 0;
  }
  return f->min_num_decode_surfaces;
}

int NvDecoder::dec_cb(void* u, CUVIDPICPARAMS* p) {
  auto* self = (NvDecoder*)u;
  return self->cv_->cuvidDecodePicture(self->dec_, p) == CUDA_SUCCESS;
}

// Map -> D2D copy into the ring on the copy stream -> host wait -> unmap, all before returning
// to the parser. The ring slot's previous consumer is awaited on the stream (no host wait for it
// unless the ring is exhausted, which means the consumer fell `ring` frames behind).
int NvDecoder::disp_cb(void* u, CUVIDPARSERDISPINFO* d) {
  auto* self = (NvDecoder*)u;
  if (self->frames - self->released_ >= self->ring_) {
    self->err_ = "decoder ring overflow (consumer did not release slots in time)";
    return 0;
  }
  int slot = (int)(self->frames % self->ring_);
  CUVIDPROCPARAMS pp{};
  pp.progressive_frame = d->progressive_frame;
  pp.top_field_first = d->top_field_first;
  pp.unpaired_field = d->repeat_first_field < 0;
  pp.output_stream = self->stream_;
  unsigned long long ptr = 0;
  unsigned int pitch = 0;
  CUresult r = self->cv_->cuvidMapVideoFrame(self->dec_, d->picture_index, &ptr, &pitch, &pp);
  if (r != CUDA_SUCCESS) {
    self->err_ = "cuvidMapVideoFrame failed: " + std::to_string((int)r);
    return 0;
  }
  const uint8_t* y = (const uint8_t*)ptr + (size_t)self->disp_y_ * pitch + self->disp_x_;
  const uint8_t* uv = (const uint8_t*)ptr + (size_t)pitch * self->surf_h_ + (self->disp_y_ / 2) * (size_t)pitch + self->disp_x_;
  const Nv12View& dst = self->slots_[slot];
  cudaError_t e = cudaSuccess;
  if (self->no_copy_) {  // NVDEC-only ceiling: skip the copy-out, its host wait and the release
    self->cv_->cuvidUnmapVideoFrame(self->dec_, ptr);
    self->pending_.push_back(slot);
    ++self->frames;
    ++self->released_;
    return 1;
  }
  if (self->ev_set_[slot]) e = cudaStreamWaitEvent(self->stream_, self->ev_[slot], 0);
  if (e == cudaSuccess)
    e = cudaMemcpy2DAsync(dst.y, dst.pitch, y, pitch, dst.w, dst.h, cudaMemcpyDeviceToDevice, self->stream_);
  if (e == cudaSuccess)
    e = cudaMemcpy2DAsync(dst.uv, dst.pitch, uv, pitch, dst.w, dst.h / 2, cudaMemcpyDeviceToDevice, self->stream_);
  if (e == cudaSuccess) e = cudaStreamSynchronize(self->stream_);
  self->cv_->cuvidUnmapVideoFrame(self->dec_, ptr);
  if (e != cudaSuccess) {
    self->err_ = std::string("frame copy: ") + cudaGetErrorString(e);
    return 0;
  }
  self->pending_.push_back(slot);
  ++self->frames;
  return 1;
}

void NvDecoder::parse(CUVIDSOURCEDATAPACKET* p) {
  CUresult r = cv_->cuvidParseVideoData(parser_, p);
  if (!err_.empty()) throw std::runtime_error("[NVDEC] " + err_);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuvidParseVideoData failed: " + std::to_string((int)r));
}

bool NvDecoder::pump() {
  if (eos_sent_) return false;
  CUVIDSOURCEDATAPACKET sp{};
  while (!eof_) {
    int r = av_read_frame(fmt_, pkt_);
    if (r < 0) {
      eof_ = true;
      av_bsf_send_packet(bsf_, nullptr);  // flush the filter
      break;
    }
    if (pkt_->stream_index != vidx_) {
      av_packet_unref(pkt_);
      continue;
    }
    if (av_bsf_send_packet(bsf_, pkt_) < 0) throw std::runtime_error("bsf send failed");
    break;
  }
  bool fed = false;
  while (av_bsf_receive_packet(bsf_, pkt_) == 0) {
    sp.payload = pkt_->data;
    sp.payload_size = pkt_->size;
    sp.flags = CUVID_PKT_TIMESTAMP;
    sp.timestamp = pkt_->pts;
    parse(&sp);
    av_packet_unref(pkt_);
    fed = true;
  }
  if (!fed && eof_) {
    sp = CUVIDSOURCEDATAPACKET{};
    sp.flags = CUVID_PKT_ENDOFSTREAM;
    parse(&sp);
    eos_sent_ = true;
  }
  return true;
}

bool NvDecoder::next(Nv12View& out, int& slot) {
  while (pending_.empty())
    if (!pump()) return false;
  slot = pending_.front();
  pending_.pop_front();
  out = slots_[slot];
  return true;
}

void NvDecoder::release(int slot, cudaStream_t consumer) {
  CK(cudaEventRecord(ev_[slot], consumer));
  ev_set_[slot] = true;
  ++released_;
}
