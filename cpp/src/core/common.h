#pragma once
// Shared error handling. cudart + driver API both run on the device's primary context.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess)                                                             \
      throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_));      \
  } while (0)

#define CU(x)                                                                          \
  do {                                                                                 \
    CUresult r_ = (x);                                                                 \
    if (r_ != CUDA_SUCCESS) {                                                          \
      const char* s_ = nullptr;                                                        \
      cuGetErrorString(r_, &s_);                                                       \
      throw std::runtime_error(std::string(#x) + ": " + (s_ ? s_ : "unknown"));       \
    }                                                                                  \
  } while (0)

#define LOG(...)                        \
  do {                                  \
    std::fprintf(stderr, __VA_ARGS__);  \
    std::fputc('\n', stderr);           \
  } while (0)
