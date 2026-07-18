# C++ server — engineering decisions

Rationale for the non-obvious calls. Extended each phase.

## Pre-normalized tensors on the wire (scope boundary)
The server accepts already-preprocessed `[B,3,224,224]` fp32 tensors, not JPEGs. Image decode/resize
is deliberately out of scope: the Python harness feeds the engine the *same* preprocessed tensors, so
benchmarking measures **batching + the TensorRT path**, not libjpeg/OpenCV. It keeps the comparison to
pytorch/ONNX/TRT/Triton honest and the component focused on the one signal — authored concurrency + the
C++/CUDA inference path. (The fused uint8→fp32 preproc kernel is promoted separately in Phase 3, behind
a flag, so it never contaminates the fair fp32 benchmark.)

## Phase 0 — the TensorRT C++ runtime
- **TensorRT 10 API, not 8.** `enqueueV3` + `setTensorAddress` (named tensors), `delete` instead of the
  removed `destroy()`, `getNbIOTensors`/`getIOTensorName`/`getTensorIOMode` for introspection. Pinned to
  TRT 10.5 to match `tritonserver:24.10-py3` so the existing engine deserializes.
- **pImpl.** `trt_engine.h` exposes only STL types, hiding `<NvInfer.h>`/`<cuda_runtime.h>` from the
  queue, server, and tests — they compile against a clean interface, and the CUDA-facing half stays in
  one translation unit.
- **Buffers allocated once, sized for max=64.** One CUDA stream, pinned host in/out buffers
  (`cudaHostAlloc`) for fast async H2D/D2H, device buffers for the full profile max. `infer()` reshapes
  per call (`setInputShape`) but never reallocates — per-request `cudaMalloc` would dominate latency.
  A pinned-buffer *pool* for concurrent in-flight batches is a Phase-4 stretch; Phase 0 is single-batch.
- **Fail loud on contract mismatch.** Construction verifies the engine's IO tensor names
  (`pixel_values` → `logits`); a wrong/renamed engine throws instead of silently producing garbage.
- **Parity, not a new number.** Phase 0's only claim is that the C++ path reproduces the Python-TRT
  logits (atol 1e-3, exact argmax) on ≥100 images — batch-1 both sides so fp16 batch numerics match.
  The latency line the binary prints is incidental (batch-1, includes H2D/D2H+sync); the real serving
  curve comes from the Phase-2 sweep.
