#pragma once
// Device helpers shared by the core and video kernels. Kept in one place so both paths use
// the same bilinear sampling as torch (verified bit-exact against F.interpolate).
#include <cuda_runtime.h>

#include <cstdint>

__device__ __forceinline__ uint8_t dfine_clamp_u8(float v) {
  return (uint8_t)fminf(fmaxf(v, 0.f), 255.f);  // truncation, like torch .to(uint8)
}

// PyTorch bilinear source index: max(0, (dst + 0.5) * scale - 0.5), then i1 = min(i0 + 1, n - 1).
__device__ __forceinline__ void dfine_src_index(int dst, float scale, int n, int& i0, int& i1,
                                                float& l1) {
  float r = fmaxf((dst + 0.5f) * scale - 0.5f, 0.f);
  i0 = (int)r;
  i1 = i0 < n - 1 ? i0 + 1 : i0;
  l1 = r - i0;
}
