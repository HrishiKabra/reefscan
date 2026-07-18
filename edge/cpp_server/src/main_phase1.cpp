// Phase 1 gate binary: hammer the BatchQueue with N images from `threads` concurrent producers and
// verify every coalesced result matches the batch-1 reference — correctness under concurrency + no
// deadlock (if it returns, it didn't hang).
//
//   reefscan_batch_test <engine.plan> <input.bin> <N> [threads=64] [max_batch=32] [max_delay_us=1000]
//
// argmax must match the reference 100% (batched fp16 may differ from batch-1 in the low bits, so we
// tolerate a small logit delta but require identical predictions).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <thread>
#include <vector>

#include "batch_queue.h"

using Logits = std::array<float, reefscan::TrtEngine::kNumClasses>;
static constexpr size_t kPer = static_cast<size_t>(reefscan::TrtEngine::kC) *
                               reefscan::TrtEngine::kH * reefscan::TrtEngine::kW;
static int argmax(const Logits& l) { return l[0] >= l[1] ? 0 : 1; }

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <engine.plan> <input.bin> <N> [threads] [max_batch] [max_delay_us]\n", argv[0]);
    return 2;
  }
  const std::string engine_path = argv[1], in_path = argv[2];
  const int N = std::stoi(argv[3]);
  const int threads = argc > 4 ? std::stoi(argv[4]) : 64;
  const int max_batch = argc > 5 ? std::stoi(argv[5]) : 32;
  const int max_delay = argc > 6 ? std::stoi(argv[6]) : 1000;

  std::ifstream fin(in_path, std::ios::binary);
  if (!fin) { std::fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }
  std::vector<float> input(static_cast<size_t>(N) * kPer);
  if (!fin.read(reinterpret_cast<char*>(input.data()),
                static_cast<std::streamsize>(input.size() * sizeof(float)))) {
    std::fprintf(stderr, "input.bin too small for N=%d\n", N);
    return 1;
  }

  try {
    reefscan::TrtEngine engine(engine_path);

    // reference: batch-1 logits, single-threaded, BEFORE the queue owns the engine
    std::vector<Logits> ref(N);
    for (int i = 0; i < N; ++i) ref[i] = engine.infer(input.data() + static_cast<size_t>(i) * kPer, 1)[0];

    std::vector<Logits> res(N);
    std::atomic<int> errors{0};
    auto t0 = std::chrono::steady_clock::now();
    {
      reefscan::BatchQueue bq(&engine, max_batch, max_delay);  // scheduler drives the engine now
      std::vector<std::thread> pool;
      for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
          for (int i = t; i < N; i += threads) {  // round-robin -> interleaved submissions
            try { res[i] = bq.submit(input.data() + static_cast<size_t>(i) * kPer); }
            catch (...) { errors.fetch_add(1); }
          }
        });
      }
      for (auto& th : pool) th.join();
    }  // bq destructor: clean shutdown + join scheduler
    auto t1 = std::chrono::steady_clock::now();

    int argmax_ok = 0;
    float maxdiff = 0.f;
    for (int i = 0; i < N; ++i) {
      if (argmax(res[i]) == argmax(ref[i])) ++argmax_ok;
      maxdiff = std::max({maxdiff, std::fabs(res[i][0] - ref[i][0]), std::fabs(res[i][1] - ref[i][1])});
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("[phase1] N=%d threads=%d max_batch=%d delay=%dus | %.1f req/s | "
                "argmax %d/%d | max|batched-ref|=%.2e | errors=%d\n",
                N, threads, max_batch, max_delay, N * 1000.0 / wall_ms,
                argmax_ok, N, maxdiff, errors.load());
    const bool pass = (argmax_ok == N) && (errors.load() == 0);
    std::printf(pass ? "[phase1] PASS — correct under concurrency, no deadlock\n"
                     : "[phase1] FAIL\n");
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[phase1] FATAL: %s\n", e.what());
    return 1;
  }
}
