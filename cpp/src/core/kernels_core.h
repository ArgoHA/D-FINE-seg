#pragma once
// Task kernels with no video dependency: image -> engine input, engine output -> detections,
// engine masks -> per-instance binary masks. Raw device pointers + a stream, so they compose
// with whatever the caller is already doing on the GPU.
#include <cuda_runtime.h>

#include <cstdint>

#include "dfine.h"

using Dets = DfineDets;
constexpr int kMaxDet = DFINE_MAX_DET;

// uint8 HWC (3 channels, `sp` bytes per row) -> float CHW RGB/255 at (in_h, in_w), bilinear
// like F.interpolate(align_corners=False). Squish resize, no letterbox (keep_ratio=False
// engines only). swap_rb=1 for BGR input.
void preprocess_image(const uint8_t* src, int sp, int sw, int sh, int swap_rb, float* dst,
                      int in_h, int in_w, cudaStream_t s);

// conf threshold + class filter + class-agnostic greedy NMS over the engine's score-sorted
// top-K. Boxes are scaled by (sx, sy) on the way out - pass the image/engine-input ratio to get
// boxes in image pixels, or 1,1 to keep them in engine-input space. class_mask=0 keeps all.
void postprocess(const void* labels, bool labels_i64, const float* boxes, const float* scores,
                 int k, float conf, uint64_t class_mask, float nms_iou, float sx, float sy,
                 Dets* out, cudaStream_t s);

// Engine masks [K, mh0, mw0] -> uint8 [min(count, cap), h, w], bilinear-upsampled in fp16,
// thresholded and zeroed outside each instance's own box. Mirrors TRTModel.process_masks +
// cleanup_masks. (bsx, bsy) scale boxes from their current space into the (w, h) grid.
void upsample_masks(const Dets* dets, const float* masks, int mh0, int mw0, int cap, int h, int w,
                    float bsx, float bsy, float thresh, uint8_t* dst, cudaStream_t s);
