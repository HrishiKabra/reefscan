"""Benchmark harness — the spine of ReefScan-Edge.

Every rung registers a variant via `benchmark(name, runtime, precision, predict_fn, ...)`.
The four correctness invariants from the spec are enforced here:
  1. GPU timing = warmup + torch.cuda.synchronize() around every measured call (device-aware:
     syncs only on CUDA, so the same harness runs on CPU locally and GPU on Colab).
  2. macro-F1 computed on the SAME fixed test set for every variant.
  3. (calibration handled in data.load_calibration — used by the int8 rungs.)
  4. batch-1 and batched throughput are separate rows, never conflated.

`predict_fn(x)` takes a batched input tensor and returns logits (torch.Tensor or np.ndarray);
this keeps the harness runtime-agnostic — pytorch / onnxruntime / tensorrt all plug in the same.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

CSV_PATH = "edge/results.csv"
MD_PATH = "edge/RESULTS.md"
FIELDS = ["name", "runtime", "precision", "device", "batch", "p50_ms", "p95_ms", "p99_ms",
          "throughput_ips", "peak_mem_mb", "macro_f1", "accuracy", "n_test"]


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _to_logits_np(out) -> np.ndarray:
    if torch.is_tensor(out):
        return out.detach().float().cpu().numpy()
    return np.asarray(out)


@torch.no_grad()
def time_latency(predict, x, device: str, warmup: int, iters: int) -> dict:
    """Invariant #1: warmup, then sync-bracketed perf_counter around each measured call."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        predict(x); _sync(device)
    lat = []
    for _ in range(iters):
        _sync(device); t0 = time.perf_counter()
        predict(x)
        _sync(device); lat.append((time.perf_counter() - t0) * 1e3)
    lat = np.array(lat)
    mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
    return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "throughput_ips": float(x.shape[0] * 1000.0 / lat.mean()), "peak_mem_mb": mem}


@torch.no_grad()
def macro_f1(predict, test_x: torch.Tensor, test_y: np.ndarray, device: str, eval_batch: int = 64) -> tuple[float, float]:
    """Invariant #2: accuracy on the SAME fixed test set, every variant."""
    from sklearn.metrics import accuracy_score, f1_score
    preds = []
    for i in range(0, len(test_y), eval_batch):
        xb = test_x[i:i + eval_batch].to(device)
        preds.append(_to_logits_np(predict(xb)).argmax(1))
    yp = np.concatenate(preds)
    return (float(f1_score(test_y, yp, labels=[0, 1], average="macro", zero_division=0)),
            float(accuracy_score(test_y, yp)))


def benchmark(name: str, runtime: str, precision: str, predict, test_x: torch.Tensor,
              test_y: np.ndarray, device: str, batch_sizes=(1, 32),
              warmup: int | None = None, iters: int | None = None) -> list[dict]:
    """Run one variant: macro-F1 once (batch-independent) + latency per batch size (invariant #4)."""
    f1, acc = macro_f1(predict, test_x, test_y, device)
    w = warmup if warmup is not None else (20 if device == "cuda" else 3)
    it = iters if iters is not None else (200 if device == "cuda" else 15)
    rows = []
    for bs in batch_sizes:
        x = test_x[:bs].to(device)  # a real input batch of this size
        lat = time_latency(predict, x, device, w, it)
        rows.append({"name": name, "runtime": runtime, "precision": precision, "device": device,
                     "batch": bs, "macro_f1": round(f1, 4), "accuracy": round(acc, 4),
                     "n_test": int(len(test_y)),
                     **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in lat.items()}})
    return rows


KEY = ("name", "runtime", "precision", "device", "batch")


def append_results(rows: list[dict], csv_path: str = CSV_PATH, md_path: str = MD_PATH) -> None:
    """Idempotent: a row with the same (name,runtime,precision,device,batch) REPLACES the prior
    one (last-wins), so re-running a rung never duplicates rows. First-seen order is preserved."""
    merged: dict = {}
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            merged[tuple(str(r.get(k)) for k in KEY)] = {k: r.get(k) for k in FIELDS}
    for r in rows:
        merged[tuple(str(r.get(k)) for k in KEY)] = {k: r.get(k) for k in FIELDS}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in merged.values():
            w.writerow(r)
    _write_md(csv_path, md_path)


def _write_md(csv_path: str, md_path: str) -> None:
    rows = list(csv.DictReader(open(csv_path)))

    def num(v, nd=2):
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "—"

    out = ["# ReefScan-Edge — benchmark results", "",
           "Same 1,565-image held-out test set for every variant. Latency = warmup + sync-bracketed.",
           "Batch-1 and batched rows are separate (never conflated).", "",
           "| runtime | precision | device | batch | p50 ms | p95 ms | p99 ms | throughput img/s | peak mem MB | macro-F1 | acc |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    END2END = ("cpp-trt", "triton")  # server rows: end-to-end latency, batch col = client concurrency
    for r in rows:
        mem = num(r.get("peak_mem_mb"), 0) if r.get("peak_mem_mb") not in ("", "None", None) else "—"
        mark = " †" if r["runtime"] in END2END else (" ‡" if r["precision"] == "int8-qat" else "")
        rt = r["runtime"] + mark  # self-carrying caveat marker
        out.append(f"| {rt} | {r['precision']} | {r.get('device','')} | {r['batch']} | {num(r['p50_ms'])} | "
                   f"{num(r['p95_ms'])} | {num(r['p99_ms'])} | {num(r['throughput_ips'], 1)} | {mem} | "
                   f"{num(r['macro_f1'], 4)} | {num(r['accuracy'], 4)} |")
    if any(r["runtime"] in END2END for r in rows):
        out += ["",
                "† **Server rows** (`cpp-trt`, `triton`) are measured **end-to-end with a native C++ load "
                "client**, and their `batch` column is **client concurrency** (the server coalesces batch-1 "
                "requests internally), so they are **not** directly comparable to the in-process runtime rows "
                "above (e.g. `tensorrt fp16`'s 2.24 ms is bare kernel time). Both serve the **same fp16 engine** "
                "on the same A6000, so their macro-F1 matches `tensorrt fp16` by construction — the comparison "
                "is about the **serving stack**, not the model.",
                "",
                "- **cpp-trt** = the hand-written C++ server (`edge/cpp_server/`), native C++ load client. "
                "**3.6 ms p50 @ concurrency-1** (after a `TCP_NODELAY` fix that removed a ~40 ms Nagle stall). "
                "See `edge/cpp_server/DECISIONS.md`.",
                "- **triton** = stock NVIDIA Triton 2.51.0 (`tensorrt_plan` backend, `dynamic_batching` "
                "pref 8/16/32 @ 1 ms), measured with the official **`perf_analyzer`** (native C++ gRPC client). "
                "Full 1→64 curve: `edge/serving/docs/perf_analyzer.csv`; reproduce via `edge/serving/RUNPOD.md`.",
                "",
                "**Honest crossover:** the hand-written server **wins at concurrency-1** (3.6 vs 5.0 ms p50 — no "
                "gRPC framing, no forced queue-delay wait), while Triton **wins under load** (1490 vs 1240 img/s "
                "and *lower* p50 at concurrency-32) — its mature dynamic batcher amortizes better, and its curve "
                "keeps climbing to ~1.68k img/s at concurrency-64. perf_analyzer's server-side breakdown "
                "attributes the concurrency-1 gap to ~1.2 ms queue-delay + ~1.1 ms gRPC on top of the ~2.2 ms "
                "fp16 kernel."]
    if any(r["precision"] == "int8-qat" for r in rows):
        out += ["",
                "‡ **`int8-qat`** = Quantization-Aware Training via NVIDIA TensorRT Model Optimizer "
                "(`edge/run_qat.py`): fake-quant (Q/DQ) inserted into the trained DINOv2-B, fine-tuned 3 "
                "epochs, exported as a QDQ ONNX, built as a TRT int8 engine (explicit quantization — no "
                "calibrator). **Measured on A6000** (like the server rows), so its latency is **not** "
                "comparable to the L4 `tensorrt` rows above; its **macro-F1 0.8996 is** the fair cross-variant "
                "number (F1 is GPU-independent) — the **best of the whole ladder**.",
                "",
                "It closes the int8-collapse arc: naive ORT int8 **0.399** → modelopt PTQ (max-calibrated) "
                "**0.610** → TRT PTQ (entropy) **0.884** → **QAT 0.900**. On a same-GPU (A6000) panel "
                "(`edge/docs/qat_speed_a6000.json`), QAT int8 is **Pareto-dominant**: 1.66 ms / 2022 img/s vs "
                "fp16 1.88 ms / 1712 img/s and PTQ-int8 2.03 ms / 1744 img/s. PTQ int8 wins **no** speed over "
                "fp16 (TRT leaves the outlier-heavy layers in fp16); QAT's Q/DQ nodes commit every matmul to "
                "int8 tensor cores. Arc + per-epoch history: `edge/docs/qat_history.json`."]
    open(md_path, "w").write("\n".join(out) + "\n")
