// Phase 3 correctness gate: the fused CUDA kernel must match a naive multi-op CPU reference
// (cast + transpose + normalize) to < 1e-5 — the C++ equivalent of the notebook's torch.allclose.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "preproc_kernel.h"

int main() {
  const int N = 4, H = 224, W = 224, C = 3;
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float std[3] = {0.229f, 0.224f, 0.225f};

  std::mt19937 rng(0);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<uint8_t> h_in(static_cast<size_t>(N) * H * W * C);
  for (auto& x : h_in) x = static_cast<uint8_t>(dist(rng));

  // naive multi-op CPU reference, [N,3,H,W]
  std::vector<float> ref(static_cast<size_t>(N) * C * H * W);
  for (int n = 0; n < N; ++n)
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        for (int c = 0; c < C; ++c) {
          float v = h_in[(((size_t)n * H + h) * W + w) * 3 + c] / 255.0f;
          v = (v - mean[c]) / std[c];
          ref[(((size_t)n * C + c) * H + h) * W + w] = v;
        }

  uint8_t* d_in = nullptr;
  float *d_out = nullptr, *d_mean = nullptr, *d_std = nullptr;
  cudaMalloc(&d_in, h_in.size());
  cudaMalloc(&d_out, ref.size() * sizeof(float));
  cudaMalloc(&d_mean, 3 * sizeof(float));
  cudaMalloc(&d_std, 3 * sizeof(float));
  cudaMemcpy(d_in, h_in.data(), h_in.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_std, std, 3 * sizeof(float), cudaMemcpyHostToDevice);

  reefscan::launch_fused_preproc(d_in, d_out, d_mean, d_std, N, H, W, 0);
  cudaDeviceSynchronize();

  std::vector<float> got(ref.size());
  cudaMemcpy(got.data(), d_out, ref.size() * sizeof(float), cudaMemcpyDeviceToHost);

  float maxd = 0.f;
  for (size_t i = 0; i < ref.size(); ++i) maxd = std::max(maxd, std::fabs(got[i] - ref[i]));
  std::printf("[kernel] fused preproc max|gpu - cpu ref| = %.2e  (N=%d, %zu elems)\n",
              maxd, N, ref.size());
  const bool ok = maxd < 1e-5f;
  std::printf(ok ? "[kernel] PASS — matches the multi-op reference\n" : "[kernel] FAIL\n");

  cudaFree(d_in); cudaFree(d_out); cudaFree(d_mean); cudaFree(d_std);
  return ok ? 0 : 1;
}
