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
PYTHONPATH=. python -m edge.run_rung3b          # Rung 3b — fp16 + int8 PTQ + TF32 control
PYTHONPATH=. python -m edge.plot_pareto         # Pareto frontier -> edge/docs/pareto.png
```
The Colab notebook `edge/colab/reefscan_edge.ipynb` runs the GPU rungs end-to-end (upload + Run all).
`results.csv` is append-by-replace: re-running a rung overwrites its rows, never duplicates them.

## Rung 3b findings (precision)
- **fp16 is the precision win, and it's lossless.** The ONNX graph cast to fp16 (keep_io_types)
  matches ORT-fp32 macro-F1 exactly (0.8861) and is the Pareto-optimal point: batch-1 **3.12 ms p95
  (3.2× over PyTorch fp32)**, **448 img/s batched (3.7×)** — using the tensor cores fp32 leaves idle.
- **The TF32 control resolved the Rung-3 mystery (and corrected my first guess).** Enabling TF32 in
  PyTorch lifted batch-32 throughput 122 → 226 img/s — landing right next to ORT-fp32's 237. So ~all of
  ORT's apparent "fp32" advantage was **TF32 tensor cores** (ORT enables them by default; PyTorch eager
  defaults TF32 *off* for matmul), not ONNX graph magic. BUT TF32 did **not** move accuracy
  (pytorch-tf32 stayed 0.8853, same as strict fp32), so the small 0.8853 → 0.8861 nudge on the ORT rows
  is ORT's fused-kernel numerics (LayerNorm/GELU), independent of TF32 — not the TF32 effect I'd
  speculated at Rung 3.
- **int8 static PTQ collapses this ViT — a documented negative.** Across naive all-op vs MatMul-only
  quantization × MinMax vs Entropy calibration, static int8 lands at ~0.43–0.48 macro-F1 (≈ majority
  class). DINOv2's heavy-tailed activation outliers get squashed by static int8 calibration. The
  viable int8 path is QAT or **TensorRT's int8** (entropy calibrator + int8 tensor-core kernels) —
  Rung 4. Recorded as a measured row, not hidden.

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
| 3b fp16 + int8 | onnxconverter-common 1.14.0 (fp16, needs numpy<2); ORT static int8 PTQ (QDQ, per-channel), calib = `load_calibration()` — Colab L4 |
