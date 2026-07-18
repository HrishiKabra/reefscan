#include "preproc_kernel.h"

namespace reefscan {

// One thread per element. Reads each uint8 once (HWC layout), writes each float32 once (CHW).
__global__ void fused_preproc_kernel(const uint8_t* __restrict__ in, float* __restrict__ out,
                                     const float* __restrict__ mean, const float* __restrict__ std,
                                     int N, int H, int W) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)N * H * W * 3;
  if (idx >= total) return;
  int c = idx % 3;          // input is N,H,W,C contiguous
  long hw = idx / 3;
  int w = hw % W;
  long nh = hw / W;
  int h = nh % H;
  int n = nh / H;
  float v = (float)in[idx] / 255.0f;
  v = (v - mean[c]) / std[c];
  long out_idx = (((long)n * 3 + c) * H + h) * W + w;  // write N,C,H,W
  out[out_idx] = v;
}

void launch_fused_preproc(const uint8_t* d_in, float* d_out, const float* d_mean, const float* d_std,
                          int N, int H, int W, cudaStream_t stream) {
  long total = (long)N * H * W * 3;
  int threads = 256;
  long blocks = (total + threads - 1) / threads;
  fused_preproc_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, d_mean, d_std, N, H, W);
}

}  // namespace reefscan
