# ReefScan-Edge — C++ / TensorRT batching inference server (`edge/cpp_server/`)

A from-scratch C++ serving layer for the ReefScan DINOv2-B TensorRT engine. It re-implements
what Triton's `dynamic_batching` does — a request queue + a scheduler that coalesces concurrent
single-image requests into server-side batches — but **we author the concurrency and the TensorRT
C++ inference path ourselves**, then benchmark it apples-to-apples against the existing Python /
ONNX / TRT / Triton numbers already in `edge/RESULTS.md`.

## Why this exists (read before writing code)

The rest of ReefScan's serving work is real ML-systems, but every line of C++ in the repo today is
a `load_inline` string inside a notebook, and the batching is a declarative Triton `.pbtxt`. This
component fills the one genuine gap: **compiled-language concurrency + a latency-critical C++/CUDA
path we own end-to-end.** The deliverable is the *code and the honest curve*, not a headline number.

**The signal is the authored concurrency + TRT C++ binding — not beating Triton.** Triton has years
of optimization (CUDA graphs, pinned-memory pools, multi-instance). If our server loses at the tail,
we report *why* (single instance, no graph capture, etc.). Do **not** tune for a vanity win or
conflate batch-1 with batched — same discipline as `harness.py`.

## Scope boundary (keep the comparison fair)

- The server accepts **already-preprocessed** `[B, 3, 224, 224]` fp32 tensors on the wire. Image
  decode / resize is **out of scope** — the Python harness feeds the engine the same preprocessed
  tensors, so measuring batching + TRT (not libjpeg) keeps the comparison honest and the scope tight.
- One model (the fp16 TRT engine), one GPU instance. No multi-model, no auth, no TLS.
- Everything must be **runnable on the same RunPod box as Triton** (`edge/serving/RUNPOD.md`). Match
  `tritonserver:24.10-py3` → **TensorRT 10.5** so the existing engine loads, or rebuild the engine on
  the box. An engine is TRT-version + GPU-arch specific — state the box/GPU in every result row.

## Engine contract (must match exactly)

From `edge/serving/model_repository/reefscan_dinov2/config.pbtxt`:
- input  `pixel_values`  fp32  dims `[3, 224, 224]` (batch implicit)
- output `logits`        fp32  dims `[2]`  → `[healthy, bleached]`
- optimization profile: **min 1 / opt 32 / max 64**. The server must reject/queue-split batches > 64.

## Layout

```
edge/cpp_server/
├── CLAUDE.md                 # this file
├── CMakeLists.txt            # links CUDA + TensorRT (nvinfer), builds server + tests + kernel
├── README.md                 # build + run + how to reproduce the results.csv row
├── DECISIONS.md              # DECISIONS.md-style rationale for the non-obvious calls
├── include/
│   ├── trt_engine.h
│   └── batch_queue.h
├── src/
│   ├── trt_engine.cpp        # TensorRT C++ runtime wrapper (the CUDA-facing half)
│   ├── batch_queue.cpp       # the concurrency centerpiece
│   └── server.cpp            # main() — HTTP endpoint over the queue
├── kernels/
│   ├── preproc_kernel.cu     # the fused normalize kernel, PROMOTED out of the notebook
│   └── test_kernel.cpp       # allclose vs reference — correctness gate for the kernel
├── bench/
│   └── bench_client.py       # concurrency sweep → writes a row into edge/results.csv
└── third_party/
    └── httplib.h             # cpp-httplib single header (github.com/yhirose/cpp-httplib)
```

## Component specs

### `trt_engine.{h,cpp}` — the TensorRT C++ path
- Create `IRuntime`, `deserializeCudaEngine` from the `.engine` bytes, one `IExecutionContext`.
- Own a CUDA stream, **pinned** host input/output buffers, and device buffers sized for `max=64`.
- `std::vector<std::array<float,2>> infer(const float* batch, int n)`:
  - `context->setInputShape("pixel_values", Dims4{n,3,224,224})`, `setTensorAddress` for both bindings.
  - async H2D (`cudaMemcpyAsync`) → `context->enqueueV3(stream)` → async D2H → `cudaStreamSynchronize`.
  - Assert `n <= 64`; assert engine binding names match the contract above (fail loud on mismatch).
- A tiny RAII wrapper + a TensorRT logger that forwards warnings. No leaks (destroy context/engine/runtime).

### `batch_queue.{h,cpp}` — the concurrency centerpiece
This is the part that carries the SWE-systems signal. Re-implement `dynamic_batching`:
- `struct Request { const float* pixels; std::promise<std::array<float,2>> result; };`
- A bounded MPMC queue guarded by `std::mutex` + `std::condition_variable` (bounded → backpressure).
- **One scheduler thread** loops: wait for ≥1 request, then drain up to `MAX_BATCH` (=32, the opt
  point) requests **or** until `MAX_DELAY_US` (=1000µs, matching the Triton config) elapses since the
  first request in the batch — whichever comes first. This is the exact latency/throughput trade the
  `.pbtxt`'s `preferred_batch_size` + `max_queue_delay_microseconds` encode; we're writing it.
- Assemble the coalesced batch into one contiguous pinned buffer, call `TrtEngine::infer` once,
  scatter each row back to its request's `promise`. Clients block on the `future`.
- Clean shutdown: a poison-pill / atomic `running` flag; join the scheduler thread.

### `server.cpp`
- cpp-httplib `POST /infer`: body = raw `3*224*224` fp32 little-endian (single image). Enqueue a
  `Request`, wait on the future, return the 2 logits as JSON (or raw bytes). `GET /health` → 200.
- Config via env/flags: `ENGINE_PATH`, `MAX_BATCH`, `MAX_DELAY_US`, `PORT`.

### `kernels/` — promote the fused kernel out of the notebook
- Move the `fused_preproc_kernel` from `serving_B_cuda_kernel.ipynb` into `preproc_kernel.cu` as a
  real translation unit (same `uint8 HWC → f32 NCHW normalize`, one read / one write per element).
- `test_kernel.cpp`: run the kernel + the naive multi-op reference on random input, assert max abs
  diff `< 1e-5` (the C++ equivalent of the notebook's `torch.allclose`). This is a **build gate**.
- Optional server wiring: an `/infer_raw` path that accepts uint8 and runs the kernel before TRT —
  only if Phase 3 is reached; keep it behind a flag so the fair fp32 benchmark is unaffected.

### `bench/bench_client.py` — reuse the existing measurement convention
- Mirror `edge/harness.py` exactly: warmup, then per-request wall-clock timing; percentiles via
  `np.percentile`; `throughput_ips = n * 1000 / mean_ms`. Sweep concurrency **1 → 8 → 16 → 32 → 64**.
- macro-F1 / accuracy: send the full 1,565-image test set through `/infer`, argmax the logits, compute
  F1 with `labels=[0,1]`. Because it's the same engine, **F1 must match the `tensorrt fp16` row**
  (parity check, not a new number — a mismatch means a bug in the C++ path, not a result).
- Append a row to `edge/results.csv` with `runtime="cpp-trt"`, `precision="fp16"`, the measured
  device (e.g. `cuda`), and `n_test=1565`, using the existing schema:
  `name,runtime,precision,device,batch,p50_ms,p95_ms,p99_ms,throughput_ips,peak_mem_mb,macro_f1,accuracy,n_test`.
  Then regenerate `edge/RESULTS.md` so the C++ server sits in the same table as pytorch/ONNX/TRT/Triton.

## Correctness invariants (non-negotiable — same ethos as `harness.py`)
1. **Logit parity:** C++ engine output `allclose` (atol 1e-3) with the Python TRT logits on ≥100 test
   images before any benchmarking. Gate for Phase 0.
2. **F1 parity:** server-produced macro-F1 == the `tensorrt fp16` row (same engine → same predictions).
3. **Batch-1 and batched are separate rows.** Never conflate.
4. **Every row states the GPU + box.** Numbers are meaningless without the machine.
5. **Kernel test green** before the kernel is used anywhere (Phase 3 gate).

## Phases (each has a gate — stop and report at each)

- **Phase 0 — TRT C++ path.** CMake links CUDA + `nvinfer`; deserialize engine; one synchronous
  batch-1 inference. **Gate:** logit parity (invariant 1). No queue, no server yet.
- **Phase 1 — batching queue.** `batch_queue` + scheduler thread + promise/future coalescing.
  **Gate:** N producer threads submit concurrently, every future returns the *correct* logits
  (verify against Phase 0), no deadlock under `--stress`.
- **Phase 2 — server + sweep.** cpp-httplib `/infer`; `bench_client.py` concurrency sweep; write the
  `cpp-trt` row(s); regenerate `RESULTS.md`. **Gate:** a full p50/p95/p99 + throughput curve exists
  next to the Triton/Python rows, and F1 parity holds.
- **Phase 3 — promote the kernel.** `preproc_kernel.cu` + `test_kernel` in the build. **Gate:** kernel
  allclose test passes; record a one-line microbench (fused vs naive) in `DECISIONS.md`.
- **Phase 4 — stretch (optional, each measured, each honest).** (a) swap the mutex queue for a
  lock-free MPMC ring buffer and re-measure the tail; (b) CUDA graph capture for the fixed opt-batch
  shape; (c) a pinned-buffer pool to kill per-request alloc. Only claim what you measure.

## Definition of done
- Real compiled `.cpp` / `.cu` / `.h` under a CMake build — **zero notebook-string C++** in this dir.
- `edge/results.csv` has a `cpp-trt` row; `edge/RESULTS.md` regenerated to include it.
- `DECISIONS.md` explains: why pre-normalized tensors on the wire; the queue design (mutex+condvar,
  the `MAX_DELAY_US` trade); the honest position vs Triton (what we beat, what we don't, and why).
- `README.md`: exact build (`cmake -B build && cmake --build build`), run, and reproduce-the-row steps,
  pinned to the RunPod box / TRT 10.5.
- Kernel correctness test passes in CI-or-local; note it in the README.

## Résumé bullet this earns (fill in the measured numbers)
> Built a multithreaded C++/TensorRT inference server for a DINOv2-B classifier — hand-wrote the
> dynamic-batching scheduler (bounded MPMC queue + condvar, request coalescing to a 1 ms delay
> window) around the TensorRT C++ runtime (pinned buffers, async H2D/D2H, CUDA streams); matched
> Triton's macro-F1 and reached __ req/s at p99 __ ms across a 1→64 concurrency sweep on __ GPU.

## Non-goals (say no to these)
- Image decode/resize in C++, multi-model serving, auth/TLS, a custom RPC framework, or "beating
  Triton" as the objective. Scope creep here dilutes the one clean signal: authored low-latency C++.