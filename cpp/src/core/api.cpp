// C API over the engine + core kernels. A model is one deserialized engine shared read-only;
// a slot is one execution context (own CUDA graph, own buffers), which is what makes
// concurrency and pipelining the caller's choice rather than ours.
#include "dfine.h"

#include <cuda_runtime.h>

#include <new>
#include <string>
#include <vector>

#include "common.h"
#include "engine.h"
#include "kernels_core.h"

namespace {
thread_local std::string g_err;
void set_err(const std::string& e) { g_err = e; }

// Guards the C boundary: no exception may cross it, every failure leaves a readable message.
template <typename F>
DfineStatus guard(F&& f) {
  try {
    return f();
  } catch (const std::exception& e) {
    set_err(e.what());
    return DFINE_ERR_CUDA;
  }
}
}  // namespace

struct DfineModel {
  TrtEngine engine;
  DfineInfo info{};
  int device = 0;
  float conf = 0.5f, nms_iou = 0.7f;
  uint64_t class_mask = 0;  // 0 = every class
  explicit DfineModel(const std::string& p) : engine(p) {}
};

struct DfineSlot {
  const DfineModel* m = nullptr;
  cudaStream_t stream = nullptr;  // used for graph capture and by the blocking path
  cudaEvent_t done = nullptr;
  TrtContext* ctx = nullptr;
  Dets* dets = nullptr;       // device
  uint8_t* stage = nullptr;   // device staging for host input
  size_t stage_cap = 0;
  int img_w = 0, img_h = 0;   // last infer, for box/mask scaling
  ~DfineSlot() {
    if (ctx) delete ctx;
    if (dets) cudaFree(dets);
    if (stage) cudaFree(stage);
    if (done) cudaEventDestroy(done);
    if (stream) cudaStreamDestroy(stream);
  }
};

const char* dfine_last_error(void) { return g_err.empty() ? "" : g_err.c_str(); }

DfineModel* dfine_open(const char* engine_path, int device) {
  if (!engine_path) {
    set_err("engine_path is NULL");
    return nullptr;
  }
  try {
    CK(cudaSetDevice(device));
    auto* m = new DfineModel(engine_path);
    const TrtEngine& e = m->engine;
    m->device = device;
    m->info.in_w = e.in_w;
    m->info.in_h = e.in_h;
    m->info.in_c = e.in_c;
    m->info.max_det = e.k;
    m->info.mask_h = e.mask_h;
    m->info.mask_w = e.mask_w;
    m->info.sem_h = e.sem_h;
    m->info.sem_w = e.sem_w;
    m->info.task = e.sem_seg ? DFINE_SEM_SEG : e.has_masks ? DFINE_SEGMENT : DFINE_DETECT;
    if (e.k > DFINE_MAX_DET) {
      delete m;
      set_err("engine top-K exceeds DFINE_MAX_DET");
      return nullptr;
    }
    return m;
  } catch (const std::exception& ex) {
    set_err(ex.what());
    return nullptr;
  }
}

void dfine_close(DfineModel* m) { delete m; }

const DfineInfo* dfine_info(const DfineModel* m) { return m ? &m->info : nullptr; }

void dfine_set_conf(DfineModel* m, float conf) {
  if (m) m->conf = conf;
}
void dfine_set_nms_iou(DfineModel* m, float iou) {
  if (m) m->nms_iou = iou;
}

DfineStatus dfine_set_classes(DfineModel* m, const int* ids, int n) {
  if (!m) return DFINE_ERR_ARG;
  uint64_t mask = 0;
  for (int i = 0; i < n && ids; ++i) {
    if (ids[i] < 0 || ids[i] >= 64) {
      set_err("class id out of range (0-63)");
      return DFINE_ERR_ARG;
    }
    mask |= 1ULL << ids[i];
  }
  m->class_mask = mask;
  return DFINE_OK;
}

DfineSlot* dfine_slot_create(const DfineModel* m) {
  if (!m) {
    set_err("model is NULL");
    return nullptr;
  }
  auto* s = new (std::nothrow) DfineSlot();
  if (!s) return nullptr;
  s->m = m;
  try {
    CK(cudaSetDevice(m->device));
    CK(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&s->done, cudaEventDisableTiming));
    CK(cudaMalloc(&s->dets, sizeof(Dets)));
    CK(cudaMemset(s->dets, 0, sizeof(Dets)));
    // Capture happens here, not on the hot path: the graph bakes in the buffer addresses.
    s->ctx = new TrtContext(const_cast<TrtEngine&>(m->engine), s->stream);
    return s;
  } catch (const std::exception& ex) {
    set_err(ex.what());
    delete s;
    return nullptr;
  }
}

void dfine_slot_destroy(DfineSlot* s) { delete s; }

const DfineDets* dfine_dets(const DfineSlot* s) { return s ? s->dets : nullptr; }
const float* dfine_masks(const DfineSlot* s) { return s && s->ctx ? s->ctx->masks : nullptr; }
const int32_t* dfine_sem(const DfineSlot* s) { return s && s->ctx ? s->ctx->sem : nullptr; }
cudaEvent_t dfine_event(const DfineSlot* s) { return s ? s->done : nullptr; }

DfineStatus dfine_infer_async(DfineModel* m, DfineSlot* s, const void* img, int w, int h,
                              int pitch, DfineFormat fmt, int on_device, cudaStream_t stream) {
  if (!m || !s || !img || w <= 0 || h <= 0) {
    set_err("bad argument to dfine_infer_async");
    return DFINE_ERR_ARG;
  }
  if (s->m != m) {
    set_err("slot belongs to a different model");
    return DFINE_ERR_ARG;
  }
  if (pitch <= 0) pitch = w * 3;
  return guard([&]() -> DfineStatus {
    const uint8_t* src = (const uint8_t*)img;
    int src_pitch = pitch;
    if (!on_device) {  // stage through device memory; pinned host makes this properly async
      size_t need = (size_t)w * h * 3;
      if (need > s->stage_cap) {
        if (s->stage) CK(cudaFree(s->stage));
        s->stage = nullptr;
        s->stage_cap = 0;
        CK(cudaMalloc(&s->stage, need));
        s->stage_cap = need;
      }
      CK(cudaMemcpy2DAsync(s->stage, (size_t)w * 3, img, (size_t)pitch, (size_t)w * 3, h,
                           cudaMemcpyHostToDevice, stream));
      src = s->stage;
      src_pitch = w * 3;
    }
    const TrtEngine& e = m->engine;
    preprocess_image(src, src_pitch, w, h, fmt == DFINE_BGR8, s->ctx->input, e.in_h, e.in_w,
                     stream);
    s->ctx->run(stream);
    if (!e.sem_seg)
      postprocess(s->ctx->labels, e.labels_i64, s->ctx->boxes, s->ctx->scores, e.k, m->conf,
                  m->class_mask, m->nms_iou, (float)w / e.in_w, (float)h / e.in_h, s->dets,
                  stream);
    s->img_w = w;
    s->img_h = h;
    CK(cudaEventRecord(s->done, stream));
    return DFINE_OK;
  });
}

DfineStatus dfine_infer(DfineModel* m, DfineSlot* s, const void* img, int w, int h, int pitch,
                        DfineFormat fmt, int on_device, DfineDets* dets) {
  DfineStatus st = dfine_infer_async(m, s, img, w, h, pitch, fmt, on_device, s ? s->stream : nullptr);
  if (st != DFINE_OK) return st;
  return guard([&]() -> DfineStatus {
    if (dets && m->engine.sem_seg) {
      set_err("sem_seg engines have no detections (use dfine_sem)");
      return DFINE_ERR_ARG;
    }
    if (dets) {
      CK(cudaMemcpyAsync(dets, s->dets, sizeof(Dets), cudaMemcpyDeviceToHost, s->stream));
    }
    CK(cudaStreamSynchronize(s->stream));
    return DFINE_OK;
  });
}

DfineStatus dfine_copy_dets(const DfineSlot* s, DfineDets* dst, cudaStream_t stream) {
  if (!s || !dst) return DFINE_ERR_ARG;
  if (s->m->engine.sem_seg) {
    set_err("sem_seg engines have no detections (use dfine_sem)");
    return DFINE_ERR_ARG;
  }
  return guard([&]() -> DfineStatus {
    CK(cudaStreamWaitEvent(stream, s->done, 0));
    CK(cudaMemcpyAsync(dst, s->dets, sizeof(Dets), cudaMemcpyDeviceToHost, stream));
    CK(cudaStreamSynchronize(stream));
    return DFINE_OK;
  });
}

DfineStatus dfine_upsample_masks(const DfineModel* m, const DfineSlot* s, uint8_t* dst, int cap,
                                 int h, int w, float thresh, cudaStream_t stream) {
  if (!m || !s || !dst || cap <= 0 || h <= 0 || w <= 0) return DFINE_ERR_ARG;
  if (!m->engine.has_masks) {
    set_err("engine has no masks (segment engines only)");
    return DFINE_ERR_ARG;
  }
  if (s->img_w <= 0) {
    set_err("no inference has run on this slot yet");
    return DFINE_ERR_ARG;
  }
  return guard([&]() -> DfineStatus {
    // Boxes are in image space; scale them into the (w, h) mask grid the caller asked for.
    upsample_masks(s->dets, s->ctx->masks, m->engine.mask_h, m->engine.mask_w, cap, h, w,
                   (float)w / s->img_w, (float)h / s->img_h, thresh, dst, stream);
    return DFINE_OK;
  });
}
