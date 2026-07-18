// Fused preprocessing kernel, promoted out of serving_B_cuda_kernel.ipynb into a real .cu.
// One pass: uint8 HWC [N,H,W,3] -> float32 NCHW [N,3,H,W] with ImageNet normalize
// (cast + /255 + transpose + (x-mean)/std), one read and one write per element.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace reefscan {

// All pointers are device pointers. mean/std are length-3 (per channel).
void launch_fused_preproc(const uint8_t* d_in, float* d_out, const float* d_mean, const float* d_std,
                          int N, int H, int W, cudaStream_t stream);

}  // namespace reefscan
