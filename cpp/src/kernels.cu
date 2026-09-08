#include "kernels.h"

#include <cuda_fp16.h>

#include <cmath>

namespace {

__device__ __forceinline__ uint8_t clamp_u8(float v) {
  return (uint8_t)fminf(fmaxf(v, 0.f), 255.f);  // truncation, like torch .to(uint8)
}

// PyTorch bilinear source index: max(0, (dst + 0.5) * scale - 0.5), then i1 = min(i0 + 1, n - 1).
__device__ __forceinline__ void src_index(int dst, float scale, int n, int& i0, int& i1,
                                          float& l1) {
  float r = fmaxf((dst + 0.5f) * scale - 0.5f, 0.f);
  i0 = (int)r;
  i1 = i0 < n - 1 ? i0 + 1 : i0;
  l1 = r - i0;
}

// One decoded pixel -> uint8 RGB exactly as PyNvVideoCodec's RGB output computes it (verified
// bit-exact on real frames): limited range, 255/219 luma and 255/224 chroma gain, kr/kb-derived
// coefficients, float math, truncating cast, nearest chroma.
__device__ __forceinline__ float3 rgb_at(const Nv12View& f, int x, int y, bool bt709) {
  // Bit-exact with PyNvVideoCodec's RGB output (verified on 8.4M decoded pixels): limited range,
  // 255/219 luma and 255/224 chroma gain, coefficients derived from kr/kb in float (evaluating
  // them in double instead is 1 ulp off on the G coefficient), nearest chroma, truncating cast.
  // __fmul_rn/__fadd_rn block FMA contraction, which the reference does not use either.
  const float kr = bt709 ? 0.2126f : 0.299f, kb = bt709 ? 0.0722f : 0.114f, c = 255.f / 224.f;
  const float rv = 2.f * (1.f - kr) * c, bu = 2.f * (1.f - kb) * c;
  const float gu = -2.f * kb * (1.f - kb) / (1.f - kr - kb) * c;
  const float gv = -2.f * kr * (1.f - kr) / (1.f - kr - kb) * c;
  const float fy = __fmul_rn(255.f / 219.f, (float)f.y[y * f.pitch + x] - 16.f);
  const uint8_t* uv = f.uv + (y >> 1) * f.pitch + (x & ~1);
  const float fu = (float)uv[0] - 128.f, fv = (float)uv[1] - 128.f;
  float r = __fadd_rn(fy, __fmul_rn(rv, fv));
  float g = __fadd_rn(__fadd_rn(fy, __fmul_rn(gu, fu)), __fmul_rn(gv, fv));
  float b = __fadd_rn(fy, __fmul_rn(bu, fu));
  return make_float3(clamp_u8(r), clamp_u8(g), clamp_u8(b));
}

__global__ void k_nv12_to_input(Nv12View src, float* dst, int in_h, int in_w, bool bt709) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= in_w || dy >= in_h) return;
  int x0, x1, y0, y1;
  float lx, ly;
  src_index(dx, (float)src.w / in_w, src.w, x0, x1, lx);
  src_index(dy, (float)src.h / in_h, src.h, y0, y1, ly);
  float3 a = rgb_at(src, x0, y0, bt709), b = rgb_at(src, x1, y0, bt709);
  float3 c = rgb_at(src, x0, y1, bt709), d = rgb_at(src, x1, y1, bt709);
  // Bit-exact with torch's CUDA upsample_bilinear2d + div_(255): nvcc contracts each
  // `w0*a + w1*b` into fma(w0, a, w1*b) (same for the outer sum) and the scalar division is a
  // multiply by the float reciprocal. Verified against the reference on real frames.
  const float h0 = 1.f - ly, h1 = ly, w0 = 1.f - lx, w1 = lx, inv = 1.0f / 255.0f;
  const int plane = in_h * in_w, o = dy * in_w + dx;
  auto lerp = [&](float p00, float p01, float p10, float p11) {
    float i0 = fmaf(w0, p00, __fmul_rn(w1, p01)), i1 = fmaf(w0, p10, __fmul_rn(w1, p11));
    return __fmul_rn(fmaf(h0, i0, __fmul_rn(h1, i1)), inv);
  };
  dst[o] = lerp(a.x, b.x, c.x, d.x);
  dst[o + plane] = lerp(a.y, b.y, c.y, d.y);
  dst[o + 2 * plane] = lerp(a.z, b.z, c.z, d.z);
}

// Generic bilinear plane resize; `ch` = interleaved channels (1 for Y, 2 for UV).
__global__ void k_resize_plane(const uint8_t* src, int sp, int sw, int sh, uint8_t* dst, int dp,
                               int dw, int dh, int ch) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dw || dy >= dh) return;
  int x0, x1, y0, y1;
  float lx, ly;
  src_index(dx, (float)sw / dw, sw, x0, x1, lx);
  src_index(dy, (float)sh / dh, sh, y0, y1, ly);
  for (int c = 0; c < ch; ++c) {
    float v = (1.f - ly) * ((1.f - lx) * src[y0 * sp + x0 * ch + c] + lx * src[y0 * sp + x1 * ch + c]) +
              ly * ((1.f - lx) * src[y1 * sp + x0 * ch + c] + lx * src[y1 * sp + x1 * ch + c]);
    dst[dy * dp + dx * ch + c] = clamp_u8(v);
  }
}

__device__ __forceinline__ float iou(const float* a, const float* b) {
  float iw = fminf(a[2], b[2]) - fmaxf(a[0], b[0]);
  float ih = fminf(a[3], b[3]) - fmaxf(a[1], b[1]);
  float inter = fmaxf(iw, 0.f) * fmaxf(ih, 0.f);
  float ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return inter / ua;
}

// One block. Candidates are already score-descending (TopK); valid ones are compacted in
// order, then greedy NMS runs over the (small) compacted set with one sync per survivor.
template <typename L>
__global__ void k_postprocess(const L* labels, const float* boxes, const float* scores, int k,
                              float conf, uint64_t class_mask, float thr, Dets* out) {
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
      for (int j = 0; j < 4; ++j) out->boxes[c * 4 + j] = sb[i * 4 + j];
      out->labels[c] = (int)labels[idx[i]];
      out->scores[c] = scores[idx[i]];
      out->src[c] = idx[i];
      ++c;
    }
    out->count = c;
  }
}

// Perimeter of a box split into 4 disjoint bands (top, bottom, then left/right between them).
struct Bands {
  int x1, x2, y1, y2, yt, yb, xl, xr;
};
__device__ __forceinline__ Bands bands(int x1, int y1, int x2, int y2, int t) {
  Bands b;
  b.x1 = x1; b.x2 = x2; b.y1 = y1; b.y2 = y2;
  b.yt = min(y1 + t, y2);
  b.yb = max(y2 - t, b.yt);
  b.xl = min(x1 + t, x2);
  b.xr = max(x2 - t, b.xl);
  return b;
}
__device__ __forceinline__ int band_pixels(const Bands& b) {
  int w = b.x2 - b.x1, mid = b.yb - b.yt;
  return (b.yt - b.y1) * w + (b.y2 - b.yb) * w + mid * ((b.xl - b.x1) + (b.x2 - b.xr));
}
// Map linear index -> (x, y) inside the perimeter bands.
__device__ __forceinline__ void band_xy(const Bands& b, int i, int& x, int& y) {
  int w = b.x2 - b.x1;
  int top = (b.yt - b.y1) * w;
  if (i < top) { y = b.y1 + i / w; x = b.x1 + i % w; return; }
  i -= top;
  int bot = (b.y2 - b.yb) * w;
  if (i < bot) { y = b.yb + i / w; x = b.x1 + i % w; return; }
  i -= bot;
  int lw = b.xl - b.x1, rw = b.x2 - b.xr, row = lw + rw;
  y = b.yt + i / row;
  int r = i % row;
  x = r < lw ? b.x1 + r : b.xr + (r - lw);
}

__global__ void k_draw_boxes(Nv12View f, const Dets* d, float sx, float sy, const uint8_t* pal,
                             int n_classes, int thick, float alpha) {
  int b = blockIdx.x;
  if (b >= d->count) return;
  // Same clamping as Annotator._draw_boxes (rint == torch.round, half-to-even).
  int x1 = (int)rintf(d->boxes[b * 4 + 0] * sx), y1 = (int)rintf(d->boxes[b * 4 + 1] * sy);
  int x2 = (int)rintf(d->boxes[b * 4 + 2] * sx), y2 = (int)rintf(d->boxes[b * 4 + 3] * sy);
  x1 = max(0, min(x1, f.w - 1));
  y1 = max(0, min(y1, f.h - 1));
  x2 = max(x1 + 1, min(x2, f.w));
  y2 = max(y1 + 1, min(y2, f.h));
  int cid = ((d->labels[b] % n_classes) + n_classes) % n_classes;
  float cy = pal[cid * 3], cu = pal[cid * 3 + 1], cv = pal[cid * 3 + 2];
  float ia = 1.f - alpha;
  Bands L = bands(x1, y1, x2, y2, thick);
  int nl = band_pixels(L);
  for (int i = threadIdx.x; i < nl; i += blockDim.x) {
    int x, y;
    band_xy(L, i, x, y);
    uint8_t* p = f.y + y * f.pitch + x;
    *p = clamp_u8(*p * ia + cy * alpha);
  }
  // Chroma: same bands in half-res coordinates.
  int tc = max(1, (thick + 1) >> 1);
  Bands C = bands(x1 >> 1, y1 >> 1, (x2 + 1) >> 1, (y2 + 1) >> 1, tc);
  int nc = band_pixels(C);
  for (int i = threadIdx.x; i < nc; i += blockDim.x) {
    int x, y;
    band_xy(C, i, x, y);
    uint8_t* p = f.uv + y * f.pitch + x * 2;
    p[0] = clamp_u8(p[0] * ia + cu * alpha);
    p[1] = clamp_u8(p[1] * ia + cv * alpha);
  }
}

// torch nearest: min(floor(dst * scale), n - 1) with scale = in / out.
__device__ __forceinline__ int nearest_index(int dst, float scale, int n) {
  return min((int)floorf(dst * scale), n - 1);
}

// One thread per mask pixel; boxes (scaled into mask space) and mask indices staged in shared
// memory. The fp16 round trips mirror process_masks: masks.half(), float accumulation, half
// output compared against the threshold.
__global__ void k_mask_owners(const Dets* d, const float* masks, int mh0, int mw0, int mh, int mw,
                              float bsx, float bsy, uint16_t* owner) {
  __shared__ float sb[kMaxDet * 4];
  __shared__ int ssrc[kMaxDet];
  const int cnt = d->count, tid = threadIdx.y * blockDim.x + threadIdx.x, nt = blockDim.x * blockDim.y;
  for (int i = tid; i < cnt * 4; i += nt) sb[i] = d->boxes[i] * ((i & 1) ? bsy : bsx);
  for (int i = tid; i < cnt; i += nt) ssrc[i] = d->src[i];
  __syncthreads();
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= mw || y >= mh) return;
  int x0, x1, y0, y1;
  float lx, ly;
  src_index(x, (float)mw0 / mw, mw0, x0, x1, lx);
  src_index(y, (float)mh0 / mh, mh0, y0, y1, ly);
  const float fx = (float)x, fy = (float)y, w0 = 1.f - lx, h0 = 1.f - ly;
  uint16_t o = 0;
  for (int i = 0; i < cnt; ++i) {
    const float* b = sb + i * 4;
    if (!(fx >= b[0] && fx < b[2] && fy >= b[1] && fy < b[3])) continue;  // cleanup_masks
    const float* m = masks + (size_t)ssrc[i] * mh0 * mw0;
    auto h = [&](int yy, int xx) { return __half2float(__float2half_rn(m[yy * mw0 + xx])); };
    float i0 = fmaf(w0, h(y0, x0), __fmul_rn(lx, h(y0, x1)));
    float i1 = fmaf(w0, h(y1, x0), __fmul_rn(lx, h(y1, x1)));
    if (__half2float(__float2half_rn(fmaf(h0, i0, __fmul_rn(ly, i1)))) >= 0.5f) {
      o = (uint16_t)(i + 1);
      break;
    }
  }
  owner[y * mw + x] = o;
}

// One thread per 2x2 luma block: blends the four Y samples and averages their blended chroma
// into the block's UV sample (Python blends RGB per pixel, then 2x2-averages chroma).
// S(x, y, a, cy, cu, cv) returns false where nothing is painted.
template <typename S>
__global__ void k_blend_blocks(Nv12View f, S s) {
  const int cx = blockIdx.x * blockDim.x + threadIdx.x, cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cx >= f.w / 2 || cy >= f.h / 2) return;
  uint8_t* uv = f.uv + cy * f.pitch + cx * 2;
  const float u0 = uv[0], v0 = uv[1];
  float su = 0.f, sv = 0.f;
  bool hit = false;
  for (int k = 0; k < 4; ++k) {
    const int x = 2 * cx + (k & 1), y = 2 * cy + (k >> 1);
    float a, py, pu, pv;
    if (s(x, y, a, py, pu, pv)) {
      hit = true;
      uint8_t* p = f.y + y * f.pitch + x;
      *p = clamp_u8(*p * (1.f - a) + py * a);
      su += u0 * (1.f - a) + pu * a;
      sv += v0 * (1.f - a) + pv * a;
    } else {
      su += u0;
      sv += v0;
    }
  }
  if (!hit) return;
  uv[0] = clamp_u8(su * 0.25f);
  uv[1] = clamp_u8(sv * 0.25f);
}

struct MaskSampler {
  const Dets* d;
  const uint16_t* owner;
  int mh, mw, fw, fh, nc;
  const uint8_t* pal;
  float body, edge;
  __device__ bool operator()(int x, int y, float& a, float& py, float& pu, float& pv) const {
    const int mx = nearest_index(x, (float)mw / fw, mw), my = nearest_index(y, (float)mh / fh, mh);
    const uint16_t o = owner[my * mw + mx];
    if (!o) return false;
    bool contour = false;  // 3x3 neighbourhood (in bounds) holds another id or background
    for (int dy = -1; dy <= 1 && !contour; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = mx + dx, ny = my + dy;
        if (nx >= 0 && ny >= 0 && nx < mw && ny < mh && owner[ny * mw + nx] != o) { contour = true; break; }
      }
    a = contour ? edge : body;
    const int cid = max(0, min(d->labels[o - 1], nc - 1));  // labels.clamp(max=n-1)
    py = pal[cid * 3]; pu = pal[cid * 3 + 1]; pv = pal[cid * 3 + 2];
    return true;
  }
};

struct SemSampler {
  const int32_t* map;
  int ih, iw, fw, fh, nc;
  const uint8_t* pal;
  uint64_t class_mask;
  float alpha;
  __device__ bool operator()(int x, int y, float& a, float& py, float& pu, float& pv) const {
    const int l = map[nearest_index(y, (float)ih / fh, ih) * iw + nearest_index(x, (float)iw / fw, iw)];
    if (l == 255 || (class_mask && !(l >= 0 && l < 64 && ((class_mask >> l) & 1)))) return false;
    a = alpha;
    if (l >= 0 && l < nc) { py = pal[l * 3]; pu = pal[l * 3 + 1]; pv = pal[l * 3 + 2]; }
    else { py = 16.f; pu = 128.f; pv = 128.f; }  // sem_seg_palette: ids past the table are black
    return true;
  }
};

}  // namespace

void nv12_to_input(const Nv12View& src, float* dst, int in_h, int in_w, bool bt709,
                   cudaStream_t s) {
  dim3 blk(32, 8), grd((in_w + 31) / 32, (in_h + 7) / 8);
  k_nv12_to_input<<<grd, blk, 0, s>>>(src, dst, in_h, in_w, bt709);
}

void nv12_resize(const Nv12View& src, const Nv12View& dst, cudaStream_t s) {
  if (src.w == dst.w && src.h == dst.h) {
    cudaMemcpy2DAsync(dst.y, dst.pitch, src.y, src.pitch, src.w, src.h, cudaMemcpyDeviceToDevice, s);
    cudaMemcpy2DAsync(dst.uv, dst.pitch, src.uv, src.pitch, src.w, src.h / 2,
                      cudaMemcpyDeviceToDevice, s);
    return;
  }
  dim3 blk(32, 8);
  dim3 gy((dst.w + 31) / 32, (dst.h + 7) / 8), gc((dst.w / 2 + 31) / 32, (dst.h / 2 + 7) / 8);
  k_resize_plane<<<gy, blk, 0, s>>>(src.y, src.pitch, src.w, src.h, dst.y, dst.pitch, dst.w, dst.h, 1);
  k_resize_plane<<<gc, blk, 0, s>>>(src.uv, src.pitch, src.w / 2, src.h / 2, dst.uv, dst.pitch,
                                    dst.w / 2, dst.h / 2, 2);
}

void postprocess(const void* labels, bool labels_i64, const float* boxes, const float* scores,
                 int k, float conf, uint64_t class_mask, float nms_iou, Dets* out, cudaStream_t s) {
  if (labels_i64)
    k_postprocess<long long><<<1, kMaxDet, 0, s>>>((const long long*)labels, boxes, scores, k, conf,
                                                   class_mask, nms_iou, out);
  else
    k_postprocess<int><<<1, kMaxDet, 0, s>>>((const int*)labels, boxes, scores, k, conf, class_mask,
                                             nms_iou, out);
}

void draw_boxes(const Nv12View& frame, const Dets* dets, float sx, float sy, const uint8_t* palette,
                int n_classes, int thick, float alpha, cudaStream_t s) {
  k_draw_boxes<<<kMaxDet, 256, 0, s>>>(frame, dets, sx, sy, palette, n_classes, thick, alpha);
}

void mask_owners(const Dets* dets, const float* masks, int mh0, int mw0, int mh, int mw, float bsx,
                 float bsy, uint16_t* owner, cudaStream_t s) {
  dim3 blk(32, 8), grd((mw + 31) / 32, (mh + 7) / 8);
  k_mask_owners<<<grd, blk, 0, s>>>(dets, masks, mh0, mw0, mh, mw, bsx, bsy, owner);
}

void draw_masks(const Nv12View& f, const Dets* dets, const uint16_t* owner, int mh, int mw,
                const uint8_t* palette, int n_classes, float body_alpha, float edge_alpha,
                cudaStream_t s) {
  dim3 blk(32, 8), grd((f.w / 2 + 31) / 32, (f.h / 2 + 7) / 8);
  k_blend_blocks<<<grd, blk, 0, s>>>(
      f, MaskSampler{dets, owner, mh, mw, f.w, f.h, n_classes, palette, body_alpha, edge_alpha});
}

void draw_sem_seg(const Nv12View& f, const int32_t* map, int ih, int iw, const uint8_t* palette,
                  int n_classes, uint64_t class_mask, float alpha, cudaStream_t s) {
  dim3 blk(32, 8), grd((f.w / 2 + 31) / 32, (f.h / 2 + 7) / 8);
  k_blend_blocks<<<grd, blk, 0, s>>>(f, SemSampler{map, ih, iw, f.w, f.h, n_classes, palette, class_mask, alpha});
}
