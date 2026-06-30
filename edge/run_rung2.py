"""Rung 2 — torch.compile (Inductor). Registers the compiled fp32 variant into the harness.

Same model, same weights, same test set — only the execution graph changes. torch.compile
traces the model, fuses ops, and (on CUDA, max-autotune) autotunes kernels + uses CUDA graphs.
Compilation is a ONE-TIME cost paid on the first call per input shape; we trigger + report it
separately so the benchmarked latency is steady-state (post-compile), not polluted by it.

Local (CPU): python -m edge.run_rung2     -> verifies the path (CPU compile is slow; this is a GPU rung).
Colab (GPU): same command on a CUDA box   -> the real Rung-2 numbers.

Escape hatch: REEFSCAN_COMPILE_MODE=default (or reduce-overhead) if max-autotune errors on a
given stack — the HF model can graph-break; any mode still produces a valid, fairer-than-eager run.
"""
from __future__ import annotations

import os
import time

import torch

from edge.data import load_test
from edge.harness import append_results, benchmark
from edge.model import load_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = os.environ.get("REEFSCAN_COMPILE_MODE", "max-autotune")
    print(f"[rung2] device={device}  compile_mode={mode}", flush=True)

    model = load_model(device=device)
    test_x, test_y = load_test()
    print(f"[rung2] test set: {tuple(test_x.shape)}  labels={len(test_y)}", flush=True)

    compiled = torch.compile(model, mode=mode)
    predict = lambda xb: compiled(xb)  # noqa: E731  (runtime-agnostic predict_fn)

    # One-time compile cost: warm the exact batch shapes we benchmark so timed calls are steady-state.
    print("[rung2] compiling (first calls per shape can take minutes)...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        for bs in (1, 32):
            compiled(test_x[:bs].to(device))
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"[rung2] compile+warm done in {time.perf_counter() - t0:.1f}s (one-time, not in latency)", flush=True)

    rows = benchmark("torch-compile", "torch.compile", "fp32", predict, test_x, test_y, device)
    append_results(rows)

    for r in rows:
        print(f"[rung2] batch={r['batch']:>2}  p50={r['p50_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  "
              f"thrpt={r['throughput_ips']:.1f} img/s  macroF1={r['macro_f1']}  acc={r['accuracy']}", flush=True)
    print("[rung2] appended to edge/results.csv + edge/RESULTS.md")


if __name__ == "__main__":
    main()
