"""Pareto frontier — macro-F1 vs latency across every GPU variant. The centerpiece figure.

Reads edge/results.csv, plots the cuda variants (apples-to-apples on one GPU): batch-1 p95
latency (the latency-critical axis) vs macro-F1, with the Pareto-optimal frontier highlighted.
A second panel shows batched throughput. int8 (CPU-measured — ORT CUDA EP can't accelerate it;
that's TensorRT/Rung 4) is annotated separately, not placed on the GPU latency axis.

Usage: PYTHONPATH=. python -m edge.plot_pareto   ->  edge/docs/pareto.png
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "edge/results.csv"
OUT_DIR = "edge/docs"
OUT_PATH = os.path.join(OUT_DIR, "pareto.png")


def _rows():
    out = []
    for r in csv.DictReader(open(CSV_PATH)):
        try:
            r["batch"] = int(r["batch"])
            for k in ("p50_ms", "p95_ms", "throughput_ips", "macro_f1"):
                r[k] = float(r[k])
        except (TypeError, ValueError):
            continue
        out.append(r)
    return out


def _frontier(points):
    """Pareto-optimal: lower latency AND higher macro-F1. Returns the frontier sorted by latency."""
    pts = sorted(points, key=lambda p: p[0])  # by latency asc
    best_f1, front = -1.0, []
    for lat, f1, label in pts:
        if f1 > best_f1:
            front.append((lat, f1, label))
            best_f1 = f1
    return front


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = _rows()
    label = lambda r: f"{r['runtime']}/{r['precision']}"

    cuda1 = [r for r in rows if r["device"] == "cuda" and r["batch"] == 1]
    cuda32 = [r for r in rows if r["device"] == "cuda" and r["batch"] == 32]
    # ORT static int8 (CPU) is the documented collapse — annotate it; TRT int8 (cuda, if present) is a
    # real GPU operating point and rides the scatter above like any other cuda variant.
    int8 = [r for r in rows if r["precision"] == "int8" and r["device"] == "cpu" and r["batch"] == 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel 1: batch-1 p95 latency vs macro-F1 (latency-critical), Pareto frontier highlighted.
    pts = [(r["p95_ms"], r["macro_f1"], label(r)) for r in cuda1]
    front = _frontier([(lat, f1, lbl) for lat, f1, lbl in pts])
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    ax1.plot(fx, fy, "-", color="#7a7a7a", lw=1.3, zorder=1, label="Pareto frontier")
    for lat, f1, lbl in pts:
        on = (lat, f1, lbl) in front
        ax1.scatter([lat], [f1], s=90, zorder=3,
                    color="#1b7837" if on else "#999999", edgecolor="black", lw=0.6)
        ax1.annotate(lbl, (lat, f1), textcoords="offset points", xytext=(7, 5), fontsize=8.5)
    ax1.set_xlabel("batch-1 p95 latency (ms)  —  lower is better")
    ax1.set_ylabel("macro-F1 (1,565-image test set)")
    ax1.set_title("Accuracy vs latency (GPU / L4, batch-1)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=8.5)
    if int8:
        ax1.annotate(f"int8 static PTQ: F1={int8[0]['macro_f1']:.3f} — COLLAPSES (ViT activation\n"
                     f"outliers); viable int8 = QAT / TensorRT entropy (Rung 4)",
                     xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8,
                     color="#762a83", va="bottom")

    # Panel 2: batched throughput (efficiency axis).
    order = sorted(cuda32, key=lambda r: r["throughput_ips"])
    names = [label(r) for r in order]
    vals = [r["throughput_ips"] for r in order]
    ax2.barh(names, vals, color="#2166ac", alpha=0.85)
    for i, v in enumerate(vals):
        ax2.text(v, i, f" {v:.0f}", va="center", fontsize=9)
    ax2.set_xlabel("throughput (img/s)  —  higher is better")
    ax2.set_title("Batched throughput (GPU / L4, batch-32)")
    ax2.grid(True, axis="x", alpha=0.25)

    fig.suptitle("ReefScan-Edge — DINOv2-B inference optimization ladder", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"[pareto] wrote {OUT_PATH}  ({len(cuda1)} batch-1 variants, frontier={[p[2] for p in front]})")


if __name__ == "__main__":
    main()
