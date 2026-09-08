// D-FINE-seg inference core: TensorRT engine + fused pre/postprocess CUDA kernels.
// No video codecs, no drawing - embed this in robotics/drone software and read the results.
//
//   DfineModel  one deserialized engine, shared and read-only once opened.
//   DfineSlot   one execution slot: its own TRT context, CUDA graph and buffers. Create one
//               per thread for concurrency, or two-three per thread to pipeline (submit frame
//               N+1 while your code still reads frame N).
//
// Results stay in DEVICE memory and are never copied to the host for you - if you only need
// boxes, dfine_copy_dets() is ~8 KB; masks stay on the GPU unless you ask for them. Same rule
// for resolution: masks come back on the engine's own grid, and dfine_upsample_masks() is
// opt-in, because a full-res upsample the caller does not want is pure waste.
//
// Engines must be exported with keep_ratio=False (the preprocess kernel squishes, it has no
// letterbox path), which is also what dfine export produces by default.
#ifndef DFINE_H
#define DFINE_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DFINE_MAX_DET 300  // engine top-K ceiling; dfine_info() reports the engine's actual K

typedef enum { DFINE_DETECT = 0, DFINE_SEGMENT = 1, DFINE_SEM_SEG = 2 } DfineTask;
typedef enum { DFINE_RGB8 = 0, DFINE_BGR8 = 1 } DfineFormat;  // uint8 HWC, 3 channels

typedef enum {
  DFINE_OK = 0,
  DFINE_ERR_ARG = -1,      // bad argument (null, out of range, wrong task)
  DFINE_ERR_CUDA = -2,     // a CUDA call failed
  DFINE_ERR_ENGINE = -3,   // engine load / execution failure
} DfineStatus;

// Detections after conf threshold, class filter and class-agnostic NMS.
// boxes are xyxy in the pixel space of the image you passed to infer.
typedef struct {
  int count;
  float boxes[DFINE_MAX_DET * 4];
  int labels[DFINE_MAX_DET];
  float scores[DFINE_MAX_DET];
  int src[DFINE_MAX_DET];  // row of dfine_masks() holding instance i's mask
} DfineDets;

typedef struct {
  int in_w, in_h, in_c;   // engine input
  int max_det;            // engine top-K
  int mask_h, mask_w;     // DFINE_SEGMENT: per-instance mask grid
  int sem_h, sem_w;       // DFINE_SEM_SEG: label-map size
  DfineTask task;         // follows the engine, not a setting
} DfineInfo;

typedef struct DfineModel DfineModel;
typedef struct DfineSlot DfineSlot;

// ---- lifecycle -------------------------------------------------------------------------
// Opens `engine_path` on CUDA device `device`. NULL on failure (see dfine_last_error).
DfineModel* dfine_open(const char* engine_path, int device);
void dfine_close(DfineModel* m);
const DfineInfo* dfine_info(const DfineModel* m);
// Last error on the calling thread; valid until this thread's next failing call.
const char* dfine_last_error(void);

// ---- tuning (applies to subsequent infer calls) -----------------------------------------
void dfine_set_conf(DfineModel* m, float conf);
void dfine_set_nms_iou(DfineModel* m, float iou);
// Keep only these class ids (0-63). NULL/n=0 restores "all classes".
DfineStatus dfine_set_classes(DfineModel* m, const int* ids, int n);

// ---- execution slots --------------------------------------------------------------------
// Creating a slot captures a CUDA graph; do it before the threads that will use slots start
// issuing other CUDA work. Slots are NOT thread-safe individually - one slot, one user.
DfineSlot* dfine_slot_create(const DfineModel* m);
void dfine_slot_destroy(DfineSlot* s);

// ---- inference ---------------------------------------------------------------------------
// img: uint8 HWC 3-channel, `pitch` bytes per row (0 = w * 3).
// on_device: 1 if `img` is a device pointer, 0 for host memory (staged through the slot).
//
// Async: queues preprocess + engine + postprocess on `stream` and returns. Read the results
// only after dfine_event(s) has fired; they stay valid until the next infer into this slot.
DfineStatus dfine_infer_async(DfineModel* m, DfineSlot* s, const void* img, int w, int h,
                              int pitch, DfineFormat fmt, int on_device, cudaStream_t stream);
// Blocking convenience: infer_async on the slot's own stream, sync, and (if `dets` is
// non-NULL, detect/segment only) copy the detections out. This is the "just give me the
// answer" path - it costs one sync per frame, so prefer the async form when throughput matters.
DfineStatus dfine_infer(DfineModel* m, DfineSlot* s, const void* img, int w, int h, int pitch,
                        DfineFormat fmt, int on_device, DfineDets* dets);

// ---- results (DEVICE pointers, valid once dfine_event has fired) --------------------------
const DfineDets* dfine_dets(const DfineSlot* s);   // detect + segment
const float* dfine_masks(const DfineSlot* s);      // segment: [max_det, mask_h, mask_w] fp32
const int32_t* dfine_sem(const DfineSlot* s);      // sem_seg: [sem_h, sem_w] int32
cudaEvent_t dfine_event(const DfineSlot* s);       // recorded after postprocess

// ---- opt-in transfers / work -------------------------------------------------------------
// Detections to host (~8 KB). Waits for the slot's work on `stream`, then copies and syncs.
DfineStatus dfine_copy_dets(const DfineSlot* s, DfineDets* dst, cudaStream_t stream);
// Per-instance masks upsampled to (h, w) and binarized at `thresh`, cropped to each box:
// dst is a device buffer of at least cap * h * w bytes, written as uint8 [min(count, cap), h, w].
// Mirrors TRTModel.process_masks + cleanup_masks (fp16 bilinear, >= threshold, box crop).
DfineStatus dfine_upsample_masks(const DfineModel* m, const DfineSlot* s, uint8_t* dst, int cap,
                                 int h, int w, float thresh, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // DFINE_H
