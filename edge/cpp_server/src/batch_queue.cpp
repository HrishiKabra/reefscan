#include "batch_queue.h"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace reefscan {

namespace {
constexpr size_t kPer = static_cast<size_t>(TrtEngine::kC) * TrtEngine::kH * TrtEngine::kW;
using Logits = std::array<float, TrtEngine::kNumClasses>;
}  // namespace

struct BatchQueue::Impl {
  TrtEngine* engine;
  int max_batch;
  int max_delay_us;
  size_t capacity;

  struct Request {
    const float* pixels;
    std::promise<Logits> result;
  };

  std::mutex mu;
  std::condition_variable not_empty;   // signaled when a request is enqueued (or on shutdown)
  std::condition_variable not_full;    // signaled when the queue drains below capacity (or shutdown)
  std::deque<Request> q;
  bool running = true;
  std::thread sched;
  std::vector<float> scratch;  // one contiguous [max_batch,3,224,224] staging buffer

  Impl(TrtEngine* e, int mb, int md, size_t cap)
      : engine(e), max_batch(mb), max_delay_us(md), capacity(cap),
        scratch(static_cast<size_t>(mb) * kPer) {
    if (mb < 1 || mb > TrtEngine::kMaxBatch)
      throw std::runtime_error("max_batch must be in [1, 64]");
    sched = std::thread([this] { loop(); });
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lk(mu);
      running = false;
    }
    not_empty.notify_all();
    not_full.notify_all();
    if (sched.joinable()) sched.join();
    // fail anything the scheduler didn't get to (shouldn't normally happen — it drains on shutdown)
    auto ex = std::make_exception_ptr(std::runtime_error("BatchQueue destroyed"));
    while (!q.empty()) {
      q.front().result.set_exception(ex);
      q.pop_front();
    }
  }

  void run_batch(std::vector<Request>& batch) {
    const int n = static_cast<int>(batch.size());
    for (int i = 0; i < n; ++i)
      std::memcpy(scratch.data() + static_cast<size_t>(i) * kPer, batch[i].pixels, kPer * sizeof(float));
    try {
      std::vector<Logits> out = engine->infer(scratch.data(), n);
      for (int i = 0; i < n; ++i) batch[i].result.set_value(out[i]);
    } catch (...) {
      std::exception_ptr ex = std::current_exception();
      for (int i = 0; i < n; ++i) batch[i].result.set_exception(ex);
    }
  }

  void loop() {
    std::vector<Request> batch;
    batch.reserve(max_batch);
    while (true) {
      std::unique_lock<std::mutex> lk(mu);
      not_empty.wait(lk, [this] { return !q.empty() || !running; });
      if (q.empty()) break;  // running == false and drained -> exit

      // Coalesce: take the first request, then keep pulling until max_batch or the delay window
      // (measured from the first request in the batch) expires — Triton's preferred_batch_size +
      // max_queue_delay_microseconds, written by hand.
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::microseconds(max_delay_us);
      batch.clear();
      while (static_cast<int>(batch.size()) < max_batch) {
        while (!q.empty() && static_cast<int>(batch.size()) < max_batch) {
          batch.push_back(std::move(q.front()));
          q.pop_front();
        }
        not_full.notify_all();  // capacity freed
        if (static_cast<int>(batch.size()) >= max_batch) break;
        // wait for more requests until the deadline (releases the lock while waiting)
        if (!not_empty.wait_until(lk, deadline, [this] { return !q.empty(); })) break;  // timed out
      }
      lk.unlock();  // run inference OUTSIDE the lock so producers keep enqueuing
      run_batch(batch);
    }
  }
};

BatchQueue::BatchQueue(TrtEngine* engine, int max_batch, int max_delay_us, size_t capacity)
    : impl_(std::make_unique<Impl>(engine, max_batch, max_delay_us, capacity)) {}

BatchQueue::~BatchQueue() = default;

std::array<float, TrtEngine::kNumClasses> BatchQueue::submit(const float* image) {
  std::promise<Logits> prom;
  std::future<Logits> fut = prom.get_future();
  {
    std::unique_lock<std::mutex> lk(impl_->mu);
    impl_->not_full.wait(lk, [this] { return impl_->q.size() < impl_->capacity || !impl_->running; });
    if (!impl_->running) throw std::runtime_error("BatchQueue is shutting down");
    impl_->q.push_back({image, std::move(prom)});
  }
  impl_->not_empty.notify_one();
  return fut.get();  // blocks until the scheduler sets the value (or rethrows on inference error)
}

}  // namespace reefscan
