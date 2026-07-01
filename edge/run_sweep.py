"""Serving curve — batch-size sweep of the winning TRT fp16 engine (the Rung-5/6 value).

The dynamic-batching decision without the Docker: sweep batch 1..64 on the TensorRT fp16 engine
and measure the latency-vs-throughput tradeoff, so you can answer the two questions a serving
system asks:
  1. What's the biggest batch I can run under a latency SLA? (batch-1 is latency-optimal; larger
     batches trade latency for throughput.)
  2. Where does throughput saturate (the "knee")? Past it you pay latency for ~no throughput.

Also reports cost-per-1k-inferences at each batch (throughput x an L4 cloud rate) — the number that
actually matters for a deployment budget. Outputs edge/docs/serving_curve.png + serving_curve.json.
GPU-only; reuses the Rung-4 fp16 engine. Run after run_rung4.

Usage (Colab, after Rung 4): PYTHONPATH=. python -m edge.run_sweep
"""
from __future__ import annotations

import json
import os

import torch

from edge.harness import time_latency
from edge.run_rung4 import FP16_PLAN, load_or_build, make_predict

OUT_DIR = "edge/docs"
PNG = os.path.join(OUT_DIR, "serving_curve.png")
JSON = os.path.join(OUT_DIR, "serving_curve.json")
BATCHES = (1, 2, 4, 8, 16, 32, 64)
# L4 on-demand ~ $0.80/hr (GCP g2-standard-4 ballpark, 2025). Configurable; only scales cost linearly.
L4_USD_PER_HR = float(os.environ.get("REEFSCAN_GPU_USD_PER_HR", "0.80"))
SLA_MS = float(os.environ.get("REEFSCAN_SLA_MS", "5.0"))  # example latency budget for the "max batch" call


def main():
    if not torch.cuda.is_available():
        raise SystemExit("[sweep] needs a CUDA GPU — run on Colab after Rung 4.")
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(FP16_PLAN):
        raise SystemExit(f"[sweep] {FP16_PLAN} missing — run `python -m edge.run_rung4` first.")

    engine = load_or_build("fp16", FP16_PLAN)
    predict = make_predict(engine)

    rows = []
    for bs in BATCHES:
        x = torch.randn(bs, 3, 224, 224, device="cuda")
        lat = time_latency(predict, x, "cuda", warmup=20, iters=200)
        thrpt = lat["throughput_ips"]
        cost_per_1k = 1000.0 / thrpt / 3600.0 * L4_USD_PER_HR  # $ for 1k images
        rows.append({"batch": bs, "p50_ms": round(lat["p50_ms"], 3), "p95_ms": round(lat["p95_ms"], 3),
                     "throughput_ips": round(thrpt, 1), "usd_per_1k": round(cost_per_1k, 6)})

    print(f"[sweep] TRT fp16 serving curve on {torch.cuda.get_device_name(0)} "
          f"(${L4_USD_PER_HR}/hr):", flush=True)
    print(f"    {'batch':>5} {'p50 ms':>8} {'p95 ms':>8} {'img/s':>9} {'$/1k img':>10}", flush=True)
    for r in rows:
        print(f"    {r['batch']:>5} {r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} "
              f"{r['throughput_ips']:>9.1f} {r['usd_per_1k']:>10.5f}", flush=True)

    peak = max(rows, key=lambda r: r["throughput_ips"])
    under_sla = [r for r in rows if r["p95_ms"] <= SLA_MS]
    best_sla = max(under_sla, key=lambda r: r["throughput_ips"]) if under_sla else None
    print(f"[sweep] peak throughput: {peak['throughput_ips']} img/s @ batch {peak['batch']} "
          f"(${peak['usd_per_1k']:.5f}/1k)", flush=True)
    if best_sla:
        print(f"[sweep] under {SLA_MS:.0f}ms p95 SLA: batch {best_sla['batch']} "
              f"-> {best_sla['throughput_ips']} img/s", flush=True)

    with open(JSON, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "usd_per_hr": L4_USD_PER_HR,
                   "sla_ms": SLA_MS, "rows": rows}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        xs = [r["throughput_ips"] for r in rows]
        ys = [r["p95_ms"] for r in rows]
        ax.plot(xs, ys, "-o", color="#2166ac", lw=1.6, zorder=2)
        for r in rows:
            ax.annotate(f"b{r['batch']}", (r["throughput_ips"], r["p95_ms"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.axhline(SLA_MS, color="#b2182b", ls="--", lw=1.1, label=f"{SLA_MS:.0f} ms p95 SLA")
        ax.set_xlabel("throughput (img/s)  —  higher is better")
        ax.set_ylabel("p95 latency (ms)  —  lower is better")
        ax.set_title("TensorRT fp16 serving curve — batch-size sweep (L4)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(PNG, dpi=140, bbox_inches="tight")
        print(f"[sweep] wrote {PNG} + {JSON}", flush=True)
    except Exception as e:  # noqa: BLE001 — plot optional, json is the artifact
        print(f"[sweep] (plot skipped: {e}); wrote {JSON}", flush=True)


if __name__ == "__main__":
    main()
