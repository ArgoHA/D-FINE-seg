#include "kernels_core.h"

#include <cuda_fp16.h>

#include "kernel_util.cuh"

namespace {

// Bit-exact with torch's CUDA upsample_bilinear2d + div_(255), the same contraction pattern as
// the NV12 path: nvcc folds each `w0*a + w1*b` into fma(w0, a, w1*b), and the scalar division
// is a multiply by the float reciprocal.
__global__ void k_preprocess_image(const uint8_t* src, int sp, int sw, int sh, int swap_rb,
                                   float* dst, int in_h, int in_w) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= in_w || dy >= in_h) return;
  int x0, x1, y0, y1;
  float lx, ly;
  dfine_src_index(dx, (float)sw / in_w, sw, x0, x1, lx);
  dfine_src_index(dy, (float)sh / in_h, sh, y0, y1, ly);
  const float h0 = 1.f - ly, h1 = ly, w0 = 1.f - lx, w1 = lx, inv = 1.0f / 255.0f;
  const int plane = in_h * in_w, o = dy * in_w + dx;
  for (int c = 0; c < 3; ++c) {
    const int sc = swap_rb ? 2 - c : c;
    auto p = [&](int yy, int xx) { return (float)src[yy * sp + xx * 3 + sc]; };
    float i0 = fmaf(w0, p(y0, x0), __fmul_rn(w1, p(y0, x1)));
    float i1 = fmaf(w0, p(y1, x0), __fmul_rn(w1, p(y1, x1)));
    dst[o + c * plane] = __fmul_rn(fmaf(h0, i0, __fmul_rn(h1, i1)), inv);
  }
}

__device__ __forceinline__ float iou(const float* a, const float* b) {
  float x1 = fmaxf(a[0], b[0]), y1 = fmaxf(a[1], b[1]);
  float x2 = fminf(a[2], b[2]), y2 = fminf(a[3], b[3]);
  float inter = fmaxf(x2 - x1, 0.f) * fmaxf(y2 - y1, 0.f);
  float ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return inter / ua;
}

// One block. Candidates are already score-descending (TopK); valid ones are compacted in
// order, then greedy NMS runs over the (small) compacted set with one sync per survivor.
// NMS runs in engine-input space; (sx, sy) is applied only to the boxes written out, so the
// suppression result does not depend on the caller's output scale.
template <typename L>
__global__ void k_postprocess(const L* labels, const float* boxes, const float* scores, int k,
                              float conf, uint64_t class_mask, float thr, float sx, float sy,
                              Dets* out) {
  __shared__ int idx[kMaxDet];
  __shared__ float sb[kMaxDet * 4];
  __shared__ int sup[kMaxDet];
  __shared__ int n;
  int t = threadIdx.x;
  if (t == 0) {
    n = 0;
    for (int i = 0; i < k; ++i) {
      long long l = (long long)labels[i];
      bool ok = scores[i] >= conf && (class_mask == 0 || (l >= 0 && l < 64 && ((class_mask >> l) & 1)));
      if (ok) idx[n++] = i;
    }
  }
  __syncthreads();
  int m = n;
  if (t < m) {
    for (int c = 0; c < 4; ++c) sb[t * 4 + c] = boxes[idx[t] * 4 + c];
    sup[t] = 0;
  }
  __syncthreads();
  for (int i = 0; i < m; ++i) {
    if (sup[i]) continue;  // uniform across the block: sup[i] is final before iteration i
    if (t > i && t < m && !sup[t] && iou(sb + i * 4, sb + t * 4) > thr) sup[t] = 1;
    __syncthreads();
  }
  if (t == 0) {
    int c = 0;
    for (int i = 0; i < m; ++i) {
      if (sup[i]) continue;
      for (int j = 0; j < 4; ++j) out->boxes[c * 4 + j] = sb[i * 4 + j] * ((j & 1) ? sy : sx);
      out->labels[c] = (int)labels[idx[i]];
      out->scores[c] = scores[idx[i]];
      out->src[c] = idx[i];
      ++c;
    }
    out->count = c;
  }
}

// One thread per output pixel per instance (instance on blockIdx.z). The fp16 round trips
// mirror process_masks: masks.half(), float accumulation, half output vs the threshold.
__global__ void k_upsample_masks(const Dets* d, const float* masks, int mh0, int mw0, int cap,
                                 int h, int w, float bsx, float bsy, float thresh, uint8_t* dst) {
  const int i = blockIdx.z;
  const int cnt = d->count < cap ? d->count : cap;
  if (i >= cnt) return;
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const float bx1 = d->boxes[i * 4] * bsx, by1 = d->boxes[i * 4 + 1] * bsy;
  const float bx2 = d->boxes[i * 4 + 2] * bsx, by2 = d->boxes[i * 4 + 3] * bsy;
  uint8_t v = 0;
  if ((float)x >= bx1 && (float)x < bx2 && (float)y >= by1 && (float)y < by2) {  // cleanup_masks
    int x0, x1, y0, y1;
    float lx, ly;
    dfine_src_index(x, (float)mw0 / w, mw0, x0, x1, lx);
    dfine_src_index(y, (float)mh0 / h, mh0, y0, y1, ly);
    const float* m = masks + (size_t)d->src[i] * mh0 * mw0;
    auto hf = [&](int yy, int xx) { return __half2float(__float2half_rn(m[yy * mw0 + xx])); };
    const float w0 = 1.f - lx, h0 = 1.f - ly;
    float i0 = fmaf(w0, hf(y0, x0), __fmul_rn(lx, hf(y0, x1)));
    float i1 = fmaf(w0, hf(y1, x0), __fmul_rn(lx, hf(y1, x1)));
    v = __half2float(__float2half_rn(fmaf(h0, i0, __fmul_rn(ly, i1)))) >= thresh ? 1 : 0;
  }
  dst[((size_t)i * h + y) * w + x] = v;
}

}  // namespace

void preprocess_image(const uint8_t* src, int sp, int sw, int sh, int swap_rb, float* dst,
                      int in_h, int in_w, cudaStream_t s) {
  dim3 blk(32, 8), grd((in_w + 31) / 32, (in_h + 7) / 8);
  k_preprocess_image<<<grd, blk, 0, s>>>(src, sp, sw, sh, swap_rb, dst, in_h, in_w);
}

void postprocess(const void* labels, bool labels_i64, const float* boxes, const float* scores,
                 int k, float conf, uint64_t class_mask, float nms_iou, float sx, float sy,
                 Dets* out, cudaStream_t s) {
  if (labels_i64)
    k_postprocess<long long><<<1, kMaxDet, 0, s>>>((const long long*)labels, boxes, scores, k, conf,
                                                   class_mask, nms_iou, sx, sy, out);
  else
    k_postprocess<int><<<1, kMaxDet, 0, s>>>((const int*)labels, boxes, scores, k, conf, class_mask,
                                             nms_iou, sx, sy, out);
}

void upsample_masks(const Dets* dets, const float* masks, int mh0, int mw0, int cap, int h, int w,
                    float bsx, float bsy, float thresh, uint8_t* dst, cudaStream_t s) {
  dim3 blk(32, 8), grd((w + 31) / 32, (h + 7) / 8, cap);
  k_upsample_masks<<<grd, blk, 0, s>>>(dets, masks, mh0, mw0, cap, h, w, bsx, bsy, thresh, dst);
}
