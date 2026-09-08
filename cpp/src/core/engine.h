#pragma once
// TensorRT engine shared by all workers; one execution context (+ CUDA graph) per worker.
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct TrtLogger : nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override;
};

class TrtEngine {
 public:
  explicit TrtEngine(const std::string& path);
  nvinfer1::ICudaEngine* engine() { return engine_.get(); }
  int in_h = 0, in_w = 0, in_c = 0, k = 0;  // k = top-K of the fused postprocessor
  int mask_h = 0, mask_w = 0;               // segment: per-instance mask grid (masks [1, K, mh, mw])
  int sem_h = 0, sem_w = 0;                 // sem_seg: fused-argmax label map (sem_seg [1, H, W])
  bool has_masks = false, sem_seg = false, labels_i64 = true;
  const char* task() const { return sem_seg ? "sem_seg" : has_masks ? "segment" : "detect"; }

 private:
  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
};

class TrtContext {
 public:
  TrtContext(TrtEngine& eng, cudaStream_t capture_stream);
  ~TrtContext();
  void run(cudaStream_t s);  // graph replay (falls back to enqueueV3 if capture failed)
  float* input = nullptr;    // [1, 3, in_h, in_w] fp32
  void* labels = nullptr;    // [1, K] int64 (or int32)
  float* boxes = nullptr;    // [1, K, 4] xyxy in input-size pixels
  float* scores = nullptr;   // [1, K]
  float* masks = nullptr;    // segment: [1, K, mask_h, mask_w] fp32 probabilities
  int32_t* sem = nullptr;    // sem_seg: [1, sem_h, sem_w] class ids

 private:
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  std::vector<void*> bufs_;
  cudaGraphExec_t graph_ = nullptr;
};
