# ReefScan-Edge — inference optimization

An optimization ladder over the trained DINOv2-B coral classifier: map the full
**accuracy–latency frontier** across runtimes (PyTorch → torch.compile → ONNX Runtime →
TensorRT fp16/int8 → Triton) on the **same 1,565-image held-out test set**. Full plan:
[`docs/V2_SPEC.md`](../docs/V2_SPEC.md).

## The harness is the spine
`harness.py` defines `benchmark(name, runtime, precision, predict_fn, …)`. Every rung just
registers a new `(runtime, precision, batch)` variant — same timing, same test set, appended
to `results.csv` + `RESULTS.md`. Correctness invariants enforced in `harness.py`:
1. latency = warmup + sync-bracketed timing (device-aware: syncs only on CUDA);
2. macro-F1 on the same fixed test set for every variant;
3. int8 uses `data.load_calibration()` (representative train subsample, not random);
4. batch-1 and batched rows are separate.

## Run
```bash
# from repo root
PYTHONPATH=. python -m edge.run_baseline        # Rung 1 — fp32 baseline (CPU locally, GPU on Colab)
PYTHONPATH=. python -m edge.run_rung2           # Rung 2 — torch.compile (GPU rung; run on Colab)
PYTHONPATH=. python -m edge.run_rung3           # Rung 3 — ONNX export + ONNX Runtime (CUDA EP)
```
The Colab notebook `edge/colab/reefscan_edge.ipynb` runs the GPU rungs end-to-end (upload + Run all).
`results.csv` is append-by-replace: re-running a rung overwrites its rows, never duplicates them.

## Phases (spec §6)
- **Weekend 1:** Rung 1 baseline (this) → Rungs 2–3 (torch.compile, ONNX) → Rung 3b (fp16 + int8 PTQ, first Pareto plot).
- **Weekend 2:** Rung 4 TensorRT (fp16 + int8) → Rung 5–6 Triton + perf_analyzer + cost/1k → profiling + Pareto report.
- **Optional:** Rung 7 distillation; CoreML / C++ TensorRT runtime.

## Pinned versions
CPU/scaffolding deps: see `requirements.txt`. GPU rung versions (CUDA / TensorRT / torch-cuda /
onnxruntime-gpu) are pinned in the per-rung Colab cell blocks and recorded here as each lands —
TensorRT/CUDA mismatches waste hours, so the GPU rungs run in the NGC containers
(`nvcr.io/nvidia/pytorch:<tag>`, `nvcr.io/nvidia/tritonserver:<tag>-py3`).

| rung | versions (filled per phase) |
|---|---|
| 1 PyTorch fp32 | torch 2.4.1 — local CPU verify + Colab **L4** (torch 2.4.1+cu121) |
| 2 torch.compile | torch 2.4.1 Inductor, `mode=max-autotune` — Colab L4 (cu121) |
| 3 ONNX Runtime | onnx 1.16.2, opset 17 (dynamic batch); onnxruntime-gpu 1.19.2 CUDA EP (cuDNN 9 via torch's bundled libs) — Colab L4 |
