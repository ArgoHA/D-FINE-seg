#include "trt_model.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "common.h"

void TrtLogger::log(Severity s, const char* m) noexcept {
  if (s <= Severity::kWARNING) LOG("[TRT] %s", m);
}

static size_t dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kFLOAT: case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    default: return 1;
  }
}

TrtEngine::TrtEngine(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open engine " + path);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) throw std::runtime_error("deserialize failed: " + path);
  int n_out = 0;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* n = engine_->getIOTensorName(i);
    auto d = engine_->getTensorShape(n);
    if (engine_->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT) {
      if (d.nbDims != 4 || d.d[0] != 1)
        throw std::runtime_error("expected static batch-1 NCHW input");
      in_c = d.d[1]; in_h = d.d[2]; in_w = d.d[3];
      continue;
    }
    ++n_out;
    std::string s = n;
    if (s == "labels") { k = d.d[1]; labels_i64 = engine_->getTensorDataType(n) == nvinfer1::DataType::kINT64; }
    if (s == "masks" && d.nbDims == 4) { has_masks = true; mask_h = d.d[2]; mask_w = d.d[3]; }
    if (n_out == 1 && d.nbDims == 3 && engine_->getTensorDataType(n) == nvinfer1::DataType::kINT32) {
      sem_seg = true; sem_h = d.d[1]; sem_w = d.d[2];  // TRTModel: single output = fused-argmax graph
    }
  }
  if (n_out != 1) sem_seg = false;
  if (!sem_seg && k == 0) throw std::runtime_error("unexpected engine IO: no labels output");
  if (has_masks && engine_->getTensorDataType("masks") != nvinfer1::DataType::kFLOAT)
    throw std::runtime_error("masks output must be fp32");
}

TrtContext::TrtContext(TrtEngine& eng, cudaStream_t s) {
  auto* e = eng.engine();
  ctx_.reset(e->createExecutionContext());
  if (!ctx_) throw std::runtime_error("createExecutionContext failed");
  for (int i = 0; i < e->getNbIOTensors(); ++i) {
    const char* n = e->getIOTensorName(i);
    auto d = e->getTensorShape(n);
    size_t el = 1;
    for (int j = 0; j < d.nbDims; ++j) el *= (size_t)d.d[j];
    void* p = nullptr;
    CK(cudaMalloc(&p, el * dtype_size(e->getTensorDataType(n))));
    CK(cudaMemset(p, 0, el * dtype_size(e->getTensorDataType(n))));
    bufs_.push_back(p);
    ctx_->setTensorAddress(n, p);
    std::string name = n;
    if (e->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT) input = (float*)p;
    else if (name == "labels") labels = p;
    else if (name == "boxes") boxes = (float*)p;
    else if (name == "scores") scores = (float*)p;
    else if (name == "masks") masks = (float*)p;
    else if (eng.sem_seg) sem = (int32_t*)p;
  }
  if (!input || (eng.sem_seg ? !sem : (!labels || !boxes || !scores || (eng.has_masks && !masks))))
    throw std::runtime_error("unexpected engine IO");
  for (int i = 0; i < 3; ++i)  // warm-up so kernels/algos are selected before capture
    if (!ctx_->enqueueV3(s)) throw std::runtime_error("enqueueV3 failed");
  CK(cudaStreamSynchronize(s));
  cudaGraph_t g = nullptr;
  if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess && ctx_->enqueueV3(s) &&
      cudaStreamEndCapture(s, &g) == cudaSuccess && cudaGraphInstantiate(&graph_, g, 0) == cudaSuccess) {
    cudaGraphDestroy(g);
  } else {
    cudaGetLastError();
    graph_ = nullptr;
    LOG("[TRT] CUDA graph capture failed, using enqueueV3");
  }
}

TrtContext::~TrtContext() {
  if (graph_) cudaGraphExecDestroy(graph_);
  for (void* p : bufs_) cudaFree(p);
}

void TrtContext::run(cudaStream_t s) {
  if (graph_) CK(cudaGraphLaunch(graph_, s));
  else if (!ctx_->enqueueV3(s)) throw std::runtime_error("enqueueV3 failed");
}
