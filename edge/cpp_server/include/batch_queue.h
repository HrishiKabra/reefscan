// Dynamic-batching queue — the hand-written equivalent of Triton's `dynamic_batching`.
// Many client threads submit single preprocessed images; one scheduler thread coalesces the
// concurrent requests into server-side batches (up to max_batch, or until max_delay_us elapses
// since the first request in the batch) and runs each batch through a single TrtEngine::infer.
//
// pImpl keeps the mutex/condvar/thread machinery out of this header.
#pragma once

#include <array>
#include <memory>

#include "trt_engine.h"

namespace reefscan {

class BatchQueue {
 public:
  // `engine` is owned by the caller and MUST outlive the queue (the scheduler thread drives it;
  // no other thread touches the engine, so its single-context usage stays thread-safe).
  // max_batch <= TrtEngine::kMaxBatch. capacity bounds the queue (backpressure: submit() blocks
  // when full). Starts the scheduler thread.
  BatchQueue(TrtEngine* engine, int max_batch = 32, int max_delay_us = 1000, size_t capacity = 1024);
  ~BatchQueue();  // clean shutdown: stop + join the scheduler, fail any stragglers.
  BatchQueue(const BatchQueue&) = delete;
  BatchQueue& operator=(const BatchQueue&) = delete;

  // Submit one preprocessed [3,224,224] fp32 image; blocks until its logits are ready and returns
  // them. Thread-safe — many threads call concurrently. `image` must stay valid until this returns
  // (it does: the call blocks). Throws if the queue is shutting down or inference failed.
  std::array<float, TrtEngine::kNumClasses> submit(const float* image);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace reefscan
