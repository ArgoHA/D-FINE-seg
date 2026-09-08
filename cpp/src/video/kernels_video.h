#pragma once
// Video-path CUDA kernels: everything stays NV12 until the encoder; RGB is only ever
// materialised per-sample inside the model-input kernel. Task kernels (preprocess from a plain
// image, postprocess, mask upsample) live in core/kernels_core.h.
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels_core.h"

struct Nv12View {  // one NV12 frame: Y plane + interleaved UV plane, same pitch
  uint8_t* y;
  uint8_t* uv;
  int pitch;  // bytes per row
  int w, h;   // visible size (even)
};

// NV12 -> float CHW RGB/255 at (in_h, in_w), bilinear like F.interpolate(align_corners=False)
// over the decoder's uint8 RGB (BT.601 or BT.709 limited range, nearest chroma). Squish resize,
// no letterbox: keep_ratio=False engines only, like TRTModel.gpu_run.
void nv12_to_input(const Nv12View& src, float* dst, int in_h, int in_w, bool bt709,
                   cudaStream_t s);
// NV12 bilinear resize (luma and chroma planes independently); memcpy when sizes match.
void nv12_resize(const Nv12View& src, const Nv12View& dst, cudaStream_t s);
// Box outlines blended into an NV12 frame; boxes scaled by (sx, sy) into frame pixels.
// palette: [n_classes][3] Y,U,V on device.
void draw_boxes(const Nv12View& frame, const Dets* dets, float sx, float sy, const uint8_t* palette,
                int n_classes, int thick, float alpha, cudaStream_t s);
// Instance masks -> owner map [mh, mw] (0 = background, else 1 + index into dets, first covering
// instance wins). The rendering-side counterpart of core's upsample_masks: one map instead of
// N per-instance masks, which is what the overlay needs and a fraction of the memory.
void mask_owners(const Dets* dets, const float* masks, int mh0, int mw0, int mh, int mw, float bsx,
                 float bsy, uint16_t* owner, cudaStream_t s);
// Annotator._draw_masks: body fill + contour (3x3 neighbourhood spans two ids) blended into the
// frame, owner map nearest-resampled to the frame size.
void draw_masks(const Nv12View& frame, const Dets* dets, const uint16_t* owner, int mh, int mw,
                const uint8_t* palette, int n_classes, float body_alpha, float edge_alpha,
                cudaStream_t s);
// Annotator._draw_sem_seg: dense label map [ih, iw] int32 nearest-resampled onto the frame and
// blended with `alpha`; ids outside class_mask (0 = all) count as ignore and stay unblended.
void draw_sem_seg(const Nv12View& frame, const int32_t* map, int ih, int iw, const uint8_t* palette,
                  int n_classes, uint64_t class_mask, float alpha, cudaStream_t s);
