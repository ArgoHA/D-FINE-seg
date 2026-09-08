#include "nvenc.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "common.h"
#include <ffnvcodec/dynlink_loader.h>

#define NVE(x)                                                                                  \
  do {                                                                                          \
    NVENCSTATUS st_ = (x);                                                                      \
    if (st_ != NV_ENC_SUCCESS) throw std::runtime_error(std::string(#x) + " -> " + std::to_string(st_)); \
  } while (0)

static GUID preset_guid(const std::string& p) {
  if (p == "P1") return NV_ENC_PRESET_P1_GUID;
  if (p == "P2") return NV_ENC_PRESET_P2_GUID;
  if (p == "P3") return NV_ENC_PRESET_P3_GUID;
  if (p == "P4") return NV_ENC_PRESET_P4_GUID;
  if (p == "P5") return NV_ENC_PRESET_P5_GUID;
  if (p == "P6") return NV_ENC_PRESET_P6_GUID;
  if (p == "P7") return NV_ENC_PRESET_P7_GUID;
  throw std::runtime_error("unknown NVENC preset " + p);
}

NvEncoder::NvEncoder(const std::string& out_path, int w, int h, double fps, CUcontext ctx,
                     CUstream stream, NvencFunctions* nv, const NvEncOpts& o)
    : stream_(stream), w_(w), h_(h) {
  uint32_t max_ver = 0;  // driver support is (major << 4) | minor, unlike NVENCAPI_VERSION
  const uint32_t hdr_ver = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
  NVE(nv->NvEncodeAPIGetMaxSupportedVersion(&max_ver));
  if (max_ver < hdr_ver)
    throw std::runtime_error("driver NVENC API " + std::to_string(max_ver >> 4) + "." +
                             std::to_string(max_ver & 0xf) + " < header " +
                             std::to_string(NVENCAPI_MAJOR_VERSION) + "." +
                             std::to_string(NVENCAPI_MINOR_VERSION) + " (vendor an older nvEncodeAPI.h)");
  fn_.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  NVE(nv->NvEncodeAPICreateInstance(&fn_));
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sp{};
  sp.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  sp.device = ctx;
  sp.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  sp.apiVersion = NVENCAPI_VERSION;
  NVE(fn_.nvEncOpenEncodeSessionEx(&sp, &enc_));

  int fps_num = (int)std::lround(fps * 1000), fps_den = 1000;
  NV_ENC_INITIALIZE_PARAMS ip{};
  NV_ENC_CONFIG cfg{};
  ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
  ip.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
  ip.presetGUID = preset_guid(o.preset);
  ip.tuningInfo = o.low_latency ? NV_ENC_TUNING_INFO_LOW_LATENCY : NV_ENC_TUNING_INFO_HIGH_QUALITY;
  ip.encodeWidth = ip.darWidth = ip.maxEncodeWidth = w;
  ip.encodeHeight = ip.darHeight = ip.maxEncodeHeight = h;
  ip.frameRateNum = fps_num;
  ip.frameRateDen = fps_den;
  ip.enablePTD = 1;
  ip.encodeConfig = &cfg;
  NV_ENC_PRESET_CONFIG pc{};
  pc.version = NV_ENC_PRESET_CONFIG_VER;
  pc.presetCfg.version = NV_ENC_CONFIG_VER;
  NVE(fn_.nvEncGetEncodePresetConfigEx(enc_, ip.encodeGUID, ip.presetGUID, ip.tuningInfo, &pc));
  std::memcpy(&cfg, &pc.presetCfg, sizeof(cfg));
  // Same touch-ups as NVIDIA's NvEncoder sample (what PyNvVideoCodec builds on).
  cfg.encodeCodecConfig.hevcConfig.idrPeriod = cfg.gopLength;
  cfg.encodeCodecConfig.hevcConfig.chromaFormatIDC = 1;
  if (o.bitrate > 0) {
    cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
    cfg.rcParams.averageBitRate = o.bitrate;
    cfg.rcParams.maxBitRate = o.bitrate * 2;
  }
  NVE(fn_.nvEncInitializeEncoder(enc_, &ip));
  frame_interval_p = cfg.frameIntervalP;
  lookahead = cfg.rcParams.lookaheadDepth;
  depth_ = cfg.frameIntervalP + cfg.rcParams.lookaheadDepth + o.extra_delay;
  if (depth_ < 1)  // depth_ is a ring modulus: 0 would divide by zero in submit()
    throw std::runtime_error("NVENC ring depth " + std::to_string(depth_) + " (--enc-delay too small)");
  reorder_ = std::max(cfg.frameIntervalP - 1, 0);  // max decode-to-display delay with B-frames

  NVE(fn_.nvEncSetIOCudaStreams(enc_, (NV_ENC_CUSTREAM_PTR)&stream_, (NV_ENC_CUSTREAM_PTR)&stream_));

  for (int i = 0; i < depth_; ++i) {
    CUdeviceptr p = 0;
    size_t pitch = 0;
    CU(cuMemAllocPitch(&p, &pitch, w, h * 3 / 2, 16));
    mem_.push_back(p);
    Nv12View v{(uint8_t*)p, (uint8_t*)p + pitch * h, (int)pitch, w, h};
    slots_.push_back(v);
    NV_ENC_REGISTER_RESOURCE rr{};
    rr.version = NV_ENC_REGISTER_RESOURCE_VER;
    rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    rr.resourceToRegister = (void*)p;
    rr.width = w;
    rr.height = h;
    rr.pitch = (uint32_t)pitch;
    rr.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    rr.bufferUsage = NV_ENC_INPUT_IMAGE;
    NVE(fn_.nvEncRegisterResource(enc_, &rr));
    reg_.push_back(rr.registeredResource);
    mapped_.push_back(nullptr);
    NV_ENC_CREATE_BITSTREAM_BUFFER cb{};
    cb.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    NVE(fn_.nvEncCreateBitstreamBuffer(enc_, &cb));
    bs_.push_back(cb.bitstreamBuffer);
  }

  // Muxer: Annex-B VPS/SPS/PPS as extradata; the mov muxer converts Annex-B packets to hvcC.
  uint8_t hdr[1024];
  uint32_t hdr_len = 0;
  NV_ENC_SEQUENCE_PARAM_PAYLOAD spp{};
  spp.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
  spp.spsppsBuffer = hdr;
  spp.inBufferSize = sizeof(hdr);
  spp.outSPSPPSPayloadSize = &hdr_len;
  NVE(fn_.nvEncGetSequenceParams(enc_, &spp));

  if (avformat_alloc_output_context2(&ofmt_, nullptr, nullptr, out_path.c_str()) < 0 || !ofmt_)
    throw std::runtime_error("cannot create muxer for " + out_path);
  st_ = avformat_new_stream(ofmt_, nullptr);
  st_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
  st_->codecpar->codec_id = AV_CODEC_ID_HEVC;
  st_->codecpar->width = w;
  st_->codecpar->height = h;
  st_->codecpar->extradata = (uint8_t*)av_mallocz(hdr_len + AV_INPUT_BUFFER_PADDING_SIZE);
  std::memcpy(st_->codecpar->extradata, hdr, hdr_len);
  st_->codecpar->extradata_size = hdr_len;
  tick_ = AVRational{fps_den, fps_num};  // one frame per tick
  st_->time_base = tick_;
  st_->avg_frame_rate = AVRational{fps_num, fps_den};
  if (avio_open(&ofmt_->pb, out_path.c_str(), AVIO_FLAG_WRITE) < 0)
    throw std::runtime_error("cannot open " + out_path);
  // Version-1 ctts (negative composition offsets): dts starts at 0, pts = display index, and
  // no edit list is needed (edit lists are stored in ms and would round a 1/fps offset).
  AVDictionary* opts = nullptr;
  av_dict_set(&opts, "movflags", "+negative_cts_offsets", 0);
  int hr = avformat_write_header(ofmt_, &opts);
  av_dict_free(&opts);
  if (hr < 0) throw std::runtime_error("write_header failed");
  pkt_ = av_packet_alloc();
}

NvEncoder::~NvEncoder() {
  try {
    if (!finished_) finish();
  } catch (const std::exception& e) {
    LOG("[NVENC] finish: %s", e.what());
  }
  for (int i = 0; i < depth_; ++i) {
    if (mapped_[i]) fn_.nvEncUnmapInputResource(enc_, mapped_[i]);
    fn_.nvEncUnregisterResource(enc_, reg_[i]);
    fn_.nvEncDestroyBitstreamBuffer(enc_, bs_[i]);
    cuMemFree(mem_[i]);
  }
  if (enc_) fn_.nvEncDestroyEncoder(enc_);
  if (ofmt_) {
    if (ofmt_->pb) avio_closep(&ofmt_->pb);
    avformat_free_context(ofmt_);
  }
  if (pkt_) av_packet_free(&pkt_);
}

void NvEncoder::submit(long frame_idx) {
  int s = (int)(frame_idx % depth_);
  NV_ENC_MAP_INPUT_RESOURCE mr{};
  mr.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
  mr.registeredResource = reg_[s];
  NVE(fn_.nvEncMapInputResource(enc_, &mr));
  mapped_[s] = mr.mappedResource;
  NV_ENC_PIC_PARAMS pp{};
  pp.version = NV_ENC_PIC_PARAMS_VER;
  pp.inputBuffer = mr.mappedResource;
  pp.bufferFmt = mr.mappedBufferFmt;
  pp.inputWidth = w_;
  pp.inputHeight = h_;
  pp.inputPitch = slots_[s].pitch;
  pp.outputBitstream = bs_[s];
  pp.inputTimeStamp = (uint64_t)frame_idx;
  pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  NVENCSTATUS r = fn_.nvEncEncodePicture(enc_, &pp);
  if (r != NV_ENC_SUCCESS && r != NV_ENC_ERR_NEED_MORE_INPUT)
    throw std::runtime_error("nvEncEncodePicture -> " + std::to_string(r));
  ++submitted_;
  while (retrieved_ < submitted_ - (depth_ - 1)) retrieve(retrieved_);
}

void NvEncoder::retrieve(long frame_idx) {
  int s = (int)(frame_idx % depth_);
  NV_ENC_LOCK_BITSTREAM lb{};
  lb.version = NV_ENC_LOCK_BITSTREAM_VER;
  lb.outputBitstream = bs_[s];
  NVE(fn_.nvEncLockBitstream(enc_, &lb));  // blocks until this picture is encoded
  bool key = lb.pictureType == NV_ENC_PIC_TYPE_IDR || lb.pictureType == NV_ENC_PIC_TYPE_I;
  std::string err;
  try {  // never leave the bitstream locked: destroying a locked buffer hangs the session
    write_packet(lb.bitstreamBufferPtr, (int)lb.bitstreamSizeInBytes, (long long)lb.outputTimeStamp, key);
  } catch (const std::exception& e) {
    err = e.what();
  }
  fn_.nvEncUnlockBitstream(enc_, bs_[s]);
  if (mapped_[s]) {
    fn_.nvEncUnmapInputResource(enc_, mapped_[s]);
    mapped_[s] = nullptr;
  }
  ++retrieved_;
  if (!err.empty()) throw std::runtime_error(err);
}

void NvEncoder::write_packet(void* data, int size, long long pts_ticks, bool key) {
  AVPacket& pkt = *pkt_;
  pkt.data = (uint8_t*)data;
  pkt.size = size;
  pkt.stream_index = st_->index;
  // Output arrives in decode order. pts = display index; dts = packet index minus the reorder
  // depth, so pts >= dts holds and the first pts is 0. The first `reorder_` dts are negative:
  // with negative_cts_offsets the muxer stores them as version-1 ctts, no edit list needed.
  pkt.dts = packets - reorder_;
  pkt.pts = pts_ticks;
  pkt.duration = 1;
  pkt.flags = key ? AV_PKT_FLAG_KEY : 0;
  av_packet_rescale_ts(&pkt, tick_, st_->time_base);
  if (av_write_frame(ofmt_, &pkt) < 0) throw std::runtime_error("av_write_frame failed");
  ++packets;
}

void NvEncoder::finish() {
  if (finished_) return;
  finished_ = true;
  NV_ENC_PIC_PARAMS pp{};
  pp.version = NV_ENC_PIC_PARAMS_VER;
  pp.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
  NVE(fn_.nvEncEncodePicture(enc_, &pp));
  while (retrieved_ < submitted_) retrieve(retrieved_);
  av_write_trailer(ofmt_);
  avio_closep(&ofmt_->pb);
}
