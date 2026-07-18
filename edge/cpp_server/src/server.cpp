// Phase 2: cpp-httplib HTTP server over the dynamic-batching queue.
//   POST /infer   body = 3*224*224 fp32 little-endian (one preprocessed image)
//                 -> 2 fp32 little-endian logits [healthy, bleached]
//   GET  /health  -> 200 "ok"
//
// Config via env: ENGINE_PATH (required), PORT (8000), MAX_BATCH (32), MAX_DELAY_US (1000).
// Image decode/resize is out of scope (see CLAUDE.md) — the wire carries preprocessed tensors so
// the benchmark measures batching + TensorRT, not libjpeg.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "batch_queue.h"
#include "httplib.h"
#include "trt_engine.h"

namespace {
int env_int(const char* k, int def) { const char* v = std::getenv(k); return v ? std::atoi(v) : def; }
}  // namespace

int main() {
  const char* engine_path = std::getenv("ENGINE_PATH");
  if (!engine_path) { std::cerr << "set ENGINE_PATH to the .plan\n"; return 2; }
  const int port = env_int("PORT", 8000);
  const int max_batch = env_int("MAX_BATCH", 32);
  const int max_delay = env_int("MAX_DELAY_US", 1000);
  constexpr size_t kBytes = static_cast<size_t>(reefscan::TrtEngine::kC) * reefscan::TrtEngine::kH *
                            reefscan::TrtEngine::kW * sizeof(float);

  reefscan::TrtEngine engine(engine_path);
  reefscan::BatchQueue queue(&engine, max_batch, max_delay);
  std::cerr << "[server] engine=" << engine_path << " max_batch=" << max_batch
            << " max_delay_us=" << max_delay << " port=" << port << "\n";

  httplib::Server svr;
  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("ok", "text/plain");
  });
  svr.Post("/infer", [&](const httplib::Request& req, httplib::Response& res) {
    if (req.body.size() != kBytes) {
      res.status = 400;
      res.set_content("expected " + std::to_string(kBytes) + " bytes (3*224*224 fp32)", "text/plain");
      return;
    }
    try {
      auto logits = queue.submit(reinterpret_cast<const float*>(req.body.data()));
      res.set_content(reinterpret_cast<const char*>(logits.data()),
                      reefscan::TrtEngine::kNumClasses * sizeof(float), "application/octet-stream");
    } catch (const std::exception& e) {
      res.status = 500;
      res.set_content(e.what(), "text/plain");
    }
  });

  std::cerr << "[server] listening on 0.0.0.0:" << port << "\n";
  if (!svr.listen("0.0.0.0", port)) { std::cerr << "[server] listen failed\n"; return 1; }
  return 0;
}
