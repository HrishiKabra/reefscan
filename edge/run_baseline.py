"""Rung 1 — PyTorch fp32 baseline (the control). Registers the fp32 variant into the harness.

Local (CPU): python -m edge.run_baseline      -> verifies macro-F1 + CPU latency.
Colab (GPU): same command on a CUDA box        -> the real GPU baseline numbers.
"""
from __future__ import annotations

import torch

from edge.data import load_test
from edge.harness import append_results, benchmark
from edge.model import load_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rung1] device={device}", flush=True)
    model = load_model(device=device)
    test_x, test_y = load_test()
    print(f"[rung1] test set: {tuple(test_x.shape)}  labels={len(test_y)}", flush=True)

    predict = lambda xb: model(xb)  # noqa: E731  (runtime-agnostic predict_fn)
    rows = benchmark("pytorch-fp32", "pytorch", "fp32", predict, test_x, test_y, device)
    append_results(rows)

    for r in rows:
        print(f"[rung1] batch={r['batch']:>2}  p50={r['p50_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  "
              f"thrpt={r['throughput_ips']:.1f} img/s  macroF1={r['macro_f1']}  acc={r['accuracy']}", flush=True)
    print("[rung1] appended to edge/results.csv + edge/RESULTS.md")


if __name__ == "__main__":
    main()
