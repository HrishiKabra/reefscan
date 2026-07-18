// TensorRT C++ runtime wrapper for the ReefScan DINOv2-B classifier engine.
// Contract (edge/serving/model_repository/reefscan_dinov2/config.pbtxt):
//   input  "pixel_values"  fp32  [B,3,224,224]
//   output "logits"        fp32  [B,2]   -> [healthy, bleached]
//   optimization profile: min 1 / opt 32 / max 64.
//
// pImpl so this header stays free of <NvInfer.h>/<cuda_runtime.h> — callers (batch_queue,
// server, tests) include only this. Phase 0: construction + a blocking infer(). No queue yet.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace reefscan {

class TrtEngine {
 public:
  static constexpr int kMaxBatch = 64;
  static constexpr int kC = 3, kH = 224, kW = 224, kNumClasses = 2;
  static constexpr const char* kInputName = "pixel_values";
  static constexpr const char* kOutputName = "logits";

  // Deserializes the serialized engine at `engine_path`, allocates the CUDA stream, pinned host
  // buffers, and device buffers sized for kMaxBatch. Throws std::runtime_error on any failure or
  // if the engine's IO tensor names don't match the contract (fail loud).
  explicit TrtEngine(const std::string& engine_path);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  // Run inference on a contiguous [n,3,224,224] fp32 host buffer; returns n rows of 2 logits.
  // n must be in [1, kMaxBatch]. Blocking (synchronizes the stream). Not thread-safe: one engine
  // is driven by one thread (the scheduler in Phase 1) — that's the intended usage.
  std::vector<std::array<float, kNumClasses>> infer(const float* input, int n);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace reefscan
