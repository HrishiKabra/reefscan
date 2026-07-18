# ReefScan-Edge — C++ / TensorRT batching inference server

A from-scratch C++ serving layer for the DINOv2-B TensorRT engine: we hand-write the TensorRT C++
inference path **and** the dynamic-batching scheduler (the thing Triton's `.pbtxt` does declaratively),
then benchmark it in the same `edge/RESULTS.md` table as pytorch/ONNX/TRT/Triton. Full plan +
rationale: [`CLAUDE.md`](CLAUDE.md). The signal is authored low-latency C++/CUDA, **not** beating Triton.

Runs on a GPU + C++ toolchain box (the RunPod box from [`../serving/RUNPOD.md`](../serving/RUNPOD.md)),
**TensorRT 10.5** to match the existing engine (`tritonserver:24.10-py3`), or rebuild the engine there.

## Status
- **Phase 0 — TRT C++ path** ✅ **GATE PASSED.** `trt_engine.{h,cpp}` (deserialize → pinned buffers →
  `enqueueV3`) + `reefscan_infer` batch-1 binary. Verified on a **RunPod RTX A6000 (secure)**,
  `nvcr.io/nvidia/pytorch:24.10-py3` (CUDA 12.6, **TensorRT 10.5**, GNU 11.4), engine built on-box:
  the C++ compiled clean and reproduced the Python-TRT logits **bit-for-bit** —
  `max|py − cpp| = 0.00e+00`, argmax **128/128**. (Batch-1 C++ path: 2.56 ms/img incl. H2D/D2H+sync;
  the real serving curve is the Phase-2 sweep.)
- Phase 1 batching queue · Phase 2 server + sweep · Phase 3 promoted kernel · Phase 4 stretch — TODO.

## Build (on the box, TensorRT 10.5)
```bash
cd edge/cpp_server
cmake -B build -DCMAKE_BUILD_TYPE=Release        # add -DTRT_ROOT=/path/to/TensorRT if not on default paths
cmake --build build -j
```
Requires: CMake ≥ 3.18, a CUDA toolkit (cudart), and TensorRT 10.5 headers + `libnvinfer`.

## Phase 0 gate — logit parity (do this before any benchmarking)
From the repo root, with the edge Python env active and the fp16 engine present
(`edge/artifacts/dinov2_trt_fp16.plan`, or `run_rung4` rebuilds it):
```bash
# 1. export 128 preprocessed test images + the Python-TRT reference logits
PYTHONPATH=. python edge/cpp_server/bench/parity_check.py --n 128
# 2. run the C++ binary over the same inputs
./edge/cpp_server/build/reefscan_infer edge/artifacts/dinov2_trt_fp16.plan \
    edge/cpp_server/_parity/input.bin edge/cpp_server/_parity/cpp_logits.bin 128
# 3. compare (PASS = max|Δ| < 1e-3 and every argmax agrees)
PYTHONPATH=. python edge/cpp_server/bench/parity_check.py --check --n 128
```
`PASS — Phase 0 gate met` means the C++ TRT runtime is bound correctly; Phase 1 (the batching queue)
builds on it. Record the GPU + box next to any number, per the invariants in `CLAUDE.md`.
