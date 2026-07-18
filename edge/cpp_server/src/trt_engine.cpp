#include "trt_engine.h"

#include <cuda_runtime.h>

#include <NvInfer.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace reefscan {
namespace {

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t _e = (call);                                                        \
    if (_e != cudaSuccess)                                                          \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
  } while (0)

// Forwards TensorRT warnings/errors to stderr; swallows info/verbose.
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
};
Logger g_logger;

std::vector<char> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open engine file: " + path);
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> buf(static_cast<size_t>(n));
  if (!f.read(buf.data(), n)) throw std::runtime_error("failed reading engine: " + path);
  return buf;
}

}  // namespace

struct TrtEngine::Impl {
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* ctx = nullptr;
  cudaStream_t stream = nullptr;
  float* d_in = nullptr;   // device [max,3,224,224]
  float* d_out = nullptr;  // device [max,2]
  float* h_in = nullptr;   // pinned host input
  float* h_out = nullptr;  // pinned host output

  ~Impl() {
    if (stream) cudaStreamDestroy(stream);
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (h_in) cudaFreeHost(h_in);
    if (h_out) cudaFreeHost(h_out);
    // TensorRT 10: interfaces are deleted with `delete` (the old destroy() is gone).
    delete ctx;
    delete engine;
    delete runtime;
  }
};

TrtEngine::TrtEngine(const std::string& engine_path) : impl_(std::make_unique<Impl>()) {
  std::vector<char> bytes = ReadFile(engine_path);

  impl_->runtime = nvinfer1::createInferRuntime(g_logger);
  if (!impl_->runtime) throw std::runtime_error("createInferRuntime failed");
  impl_->engine = impl_->runtime->deserializeCudaEngine(bytes.data(), bytes.size());
  if (!impl_->engine) throw std::runtime_error("deserializeCudaEngine failed (TRT version / GPU arch mismatch?)");
  impl_->ctx = impl_->engine->createExecutionContext();
  if (!impl_->ctx) throw std::runtime_error("createExecutionContext failed");

  // Verify the IO tensor names against the contract — fail loud if the engine isn't ours.
  bool has_in = false, has_out = false;
  for (int i = 0; i < impl_->engine->getNbIOTensors(); ++i) {
    const char* name = impl_->engine->getIOTensorName(i);
    const auto mode = impl_->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT && std::strcmp(name, kInputName) == 0) has_in = true;
    if (mode == nvinfer1::TensorIOMode::kOUTPUT && std::strcmp(name, kOutputName) == 0) has_out = true;
  }
  if (!has_in || !has_out)
    throw std::runtime_error("engine IO names don't match contract (expected pixel_values -> logits)");

  CUDA_CHECK(cudaStreamCreate(&impl_->stream));
  const size_t in_elems = static_cast<size_t>(kMaxBatch) * kC * kH * kW;
  const size_t out_elems = static_cast<size_t>(kMaxBatch) * kNumClasses;
  CUDA_CHECK(cudaMalloc(&impl_->d_in, in_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&impl_->d_out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaHostAlloc(&impl_->h_in, in_elems * sizeof(float), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&impl_->h_out, out_elems * sizeof(float), cudaHostAllocDefault));
}

TrtEngine::~TrtEngine() = default;

std::vector<std::array<float, TrtEngine::kNumClasses>> TrtEngine::infer(const float* input, int n) {
  if (n < 1 || n > kMaxBatch)
    throw std::runtime_error("batch size " + std::to_string(n) + " out of [1," + std::to_string(kMaxBatch) + "]");

  const size_t in_elems = static_cast<size_t>(n) * kC * kH * kW;
  const size_t out_elems = static_cast<size_t>(n) * kNumClasses;
  std::memcpy(impl_->h_in, input, in_elems * sizeof(float));

  if (!impl_->ctx->setInputShape(kInputName, nvinfer1::Dims4{n, kC, kH, kW}))
    throw std::runtime_error("setInputShape failed (batch out of the engine's 1..64 profile?)");
  impl_->ctx->setTensorAddress(kInputName, impl_->d_in);
  impl_->ctx->setTensorAddress(kOutputName, impl_->d_out);

  CUDA_CHECK(cudaMemcpyAsync(impl_->d_in, impl_->h_in, in_elems * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream));
  if (!impl_->ctx->enqueueV3(impl_->stream)) throw std::runtime_error("enqueueV3 failed");
  CUDA_CHECK(cudaMemcpyAsync(impl_->h_out, impl_->d_out, out_elems * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

  std::vector<std::array<float, kNumClasses>> out(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    out[i][0] = impl_->h_out[static_cast<size_t>(i) * kNumClasses + 0];
    out[i][1] = impl_->h_out[static_cast<size_t>(i) * kNumClasses + 1];
  }
  return out;
}

}  // namespace reefscan
