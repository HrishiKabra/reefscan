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

## Phase 1 — the batching queue
- **One scheduler thread owns the engine.** `TrtEngine` has a single execution context and isn't
  thread-safe; rather than lock around inference, exactly one thread (the scheduler) ever calls
  `infer()`. Producer threads only touch the queue. So there's no engine-level contention and the
  single-context usage stays correct — the concurrency lives entirely in the queue.
- **Mutex + two condvars, bounded deque.** `not_empty` wakes the scheduler; `not_full` gives
  backpressure (submit blocks when the queue is full) instead of unbounded memory growth. Inference
  runs *outside* the lock so producers keep enqueuing while a batch is on the GPU. `promise`/`future`
  hands each result back to its caller; `submit()` blocks on the future, which also keeps the caller's
  input buffer alive for free (no copy on submit — the scheduler copies into the staging buffer).
- **The delay window is the whole point.** The scheduler drains up to `max_batch` OR until
  `max_delay_us` since the batch's first request — literally Triton's `preferred_batch_size` +
  `max_queue_delay_microseconds`, written by hand. Verified argmax-identical to the batch-1 reference
  under 64–128 concurrent producers, no deadlock.

## Phase 2 — the server row, and the honest `batch` column
- **cpp-httplib, header-only.** No RPC framework, no auth/TLS (non-goals). `POST /infer` takes a raw
  preprocessed tensor and returns raw logits — minimal surface, all the interesting work is the queue.
- **The `cpp-trt` rows' `batch` column means client concurrency, not inference batch size.** Every
  other row's `batch` is the tensor batch fed to the engine; the server always receives batch-1
  requests and *coalesces them dynamically*, so the meaningful knob is concurrency. Recording it in the
  `batch` column keeps the row in the same table for a throughput comparison, but it's a different axis
  — stated here so the comparison isn't misread. `throughput_ips` is directly comparable; the latency
  columns are end-to-end HTTP (network + queue + TRT), not a bare kernel time.
- **F1 is a parity check, not a new result.** Same engine as `tensorrt fp16`, so macro-F1 must match it
  — a mismatch means a bug in the wire/queue path, not a finding.

## Phase 3 — promoting the kernel
- Same fused `uint8 HWC → fp32 NCHW → normalize` as `serving_B_cuda_kernel.ipynb`, now a real `.cu`
  translation unit compiled by nvcc under CMake, gated by an allclose-vs-multi-op-CPU-reference test
  (< 1e-5) — the C++ equivalent of the notebook's `torch.allclose`. Kept behind its own target so a
  kernel regression fails the build, and out of the fair fp32 server benchmark (the wire is already
  preprocessed) — wiring it into an optional `/infer_raw` uint8 path is future work.

## Position vs Triton (what we beat, what we don't)
This is not built to beat Triton, and the README/results say so. Triton has CUDA-graph capture, pinned
memory pools, and multi-instance execution we haven't written. If our tail latency loses at high
concurrency, that's why — single instance, mutex queue, no graph capture. The signal is the **authored**
concurrency + TensorRT C++ path, matched to Triton's macro-F1 on the same engine.

## Phase 4 — measured-future stretch (only claim what's measured)
Not yet implemented; each is a *measured* experiment when done, per the same discipline as `harness.py`:
(a) swap the mutex deque for a lock-free MPMC ring buffer and re-measure the tail; (b) CUDA-graph
capture for the fixed opt-batch shape; (c) a pinned-buffer pool to kill per-request staging copies.
Listed as future work rather than claimed.
