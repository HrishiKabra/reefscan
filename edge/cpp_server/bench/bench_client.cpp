// A native C++ load client for reefscan_server — no Python/httpx overhead, so it measures the
// SERVER, not the client. N worker threads (each its own keep-alive httplib::Client) hammer
// POST /infer; we sweep concurrency and report p50/p95/p99 + throughput. This is the honest
// throughput number the Python bench_client couldn't reach (it was client-bound).
//
//   reefscan_bench_client <host> <port> <input.bin> <n_images> [reqs_per_level=4000]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"

static constexpr size_t kPer = static_cast<size_t>(3) * 224 * 224;
static double pct(std::vector<double>& v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  double k = (v.size() - 1) * p / 100.0;
  size_t f = static_cast<size_t>(k);
  return f + 1 < v.size() ? v[f] + (v[f + 1] - v[f]) * (k - f) : v[f];
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <host> <port> <input.bin> <n_images> [reqs_per_level]\n", argv[0]);
    return 2;
  }
  const std::string host = argv[1];
  const int port = std::stoi(argv[2]);
  const int n_img = std::stoi(argv[4]);
  const int reqs = argc > 5 ? std::stoi(argv[5]) : 4000;

  std::ifstream fin(argv[3], std::ios::binary);
  std::vector<std::string> bodies(n_img);
  for (int i = 0; i < n_img; ++i) {
    bodies[i].resize(kPer * sizeof(float));
    if (!fin.read(bodies[i].data(), static_cast<std::streamsize>(kPer * sizeof(float)))) {
      std::fprintf(stderr, "input.bin too small for n_images=%d\n", n_img);
      return 1;
    }
  }

  std::printf("[cpp-client] host=%s:%d  %d reqs/level\n", host.c_str(), port, reqs);
  std::printf("    %4s %9s %9s %9s %10s\n", "conc", "p50 ms", "p95 ms", "p99 ms", "req/s");
  for (int C : {1, 8, 16, 32, 64}) {
    std::atomic<int> next{0};
    std::atomic<int> fails{0};
    std::vector<std::vector<double>> per_thread(C);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < C; ++t) {
      pool.emplace_back([&, t] {
        httplib::Client cli(host, port);
        cli.set_keep_alive(true);
        cli.set_read_timeout(60, 0);
        int i;
        while ((i = next.fetch_add(1)) < reqs) {
          const std::string& b = bodies[i % n_img];
          auto s = std::chrono::steady_clock::now();
          auto r = cli.Post("/infer", b, "application/octet-stream");
          auto e = std::chrono::steady_clock::now();
          if (r && r->status == 200) per_thread[t].push_back(std::chrono::duration<double, std::milli>(e - s).count());
          else fails.fetch_add(1);
        }
      });
    }
    for (auto& th : pool) th.join();
    auto t1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();

    std::vector<double> lat;
    for (auto& v : per_thread) lat.insert(lat.end(), v.begin(), v.end());
    std::printf("    %4d %9.2f %9.2f %9.2f %10.1f%s\n", C, pct(lat, 50), pct(lat, 95), pct(lat, 99),
                lat.size() / wall, fails.load() ? "  (had failures!)" : "");
  }
  return 0;
}
