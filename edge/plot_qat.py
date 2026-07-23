"""QAT figure — the int8 story on ONE same-GPU (A6000) canvas. Purpose-built, not the L4 pareto.

Left  : accuracy vs batch-1 latency (A6000) — fp16, int8-PTQ, int8-QAT, and the fp16-finetuned
        CONTROL (same 3-epoch fine-tune, no quantization) — showing QAT int8 is Pareto-dominant.
Right : the int8-collapse arc — naive ORT 0.399 -> modelopt PTQ 0.610 -> TRT PTQ 0.884 -> QAT 0.900,
        against the fp16 reference line.

Reads edge/docs/{qat_speed_a6000.json, qat_history.json, qat_control.json (optional)}.
Usage: PYTHONPATH=. python -m edge.plot_qat   ->  edge/docs/qat.png
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCS = "edge/docs"
OUT = os.path.join(DOCS, "qat.png")


def main():
    speed = json.load(open(f"{DOCS}/qat_speed_a6000.json"))["variants"]
    ctrl = None
    if os.path.exists(f"{DOCS}/qat_control.json"):
        ctrl = json.load(open(f"{DOCS}/qat_control.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Panel 1: accuracy vs batch-1 latency (A6000) ---
    pts = [
        ("fp16",          speed["tensorrt-fp16"]["batch1"]["p50_ms"],     speed["tensorrt-fp16"]["macro_f1"],     "#2166ac", False),
        ("int8 PTQ",      speed["tensorrt-int8-ptq"]["batch1"]["p50_ms"], speed["tensorrt-int8-ptq"]["macro_f1"], "#999999", False),
        ("int8 QAT",      speed["tensorrt-int8-qat"]["batch1"]["p50_ms"], speed["tensorrt-int8-qat"]["macro_f1"], "#1b7837", True),
    ]
    if ctrl:
        # fp16-finetuned runs at fp16 latency (same arch, retrained) — plot at fp16's p50.
        pts.append(("fp16 finetuned\n(control)", speed["tensorrt-fp16"]["batch1"]["p50_ms"],
                    ctrl["ft_control_best_test_f1"], "#7a5195", False))

    for lbl, lat, f1, color, win in pts:
        ax1.scatter([lat], [f1], s=150 if win else 95, color=color, edgecolor="black",
                    lw=1.1 if win else 0.6, zorder=3)
        dy = 8 if "control" not in lbl else -20
        ax1.annotate(lbl, (lat, f1), textcoords="offset points", xytext=(8, dy), fontsize=9,
                     fontweight="bold" if win else "normal")
    ax1.annotate("Pareto-dominant:\nfaster AND more accurate", (pts[2][1], pts[2][2]),
                 textcoords="offset points", xytext=(14, -34), fontsize=8.5, color="#1b7837")
    ax1.set_xlabel("batch-1 p50 latency (ms)  —  lower is better")
    ax1.set_ylabel("macro-F1 (1,565-image test set)")
    ax1.set_title("Accuracy vs latency — same GPU (A6000)")
    ax1.grid(True, alpha=0.25)
    ax1.margins(0.28)

    # --- Panel 2: the int8-collapse arc ---
    arc = [("ORT int8\n(naive)", 0.399, "#762a83"),
           ("modelopt PTQ\n(max-calib)", 0.610, "#c2649a"),
           ("TRT PTQ\n(entropy)", 0.884, "#4393c3"),
           ("TRT QAT\n(this)", 0.8996, "#1b7837")]
    xs = list(range(len(arc)))
    ys = [a[1] for a in arc]
    ax2.plot(xs, ys, "-o", color="#555555", lw=1.5, zorder=1)
    for x, (lbl, y, c) in zip(xs, arc):
        ax2.scatter([x], [y], s=150, color=c, edgecolor="black", lw=0.8, zorder=3)
        ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(0.889, ls="--", color="#2166ac", lw=1.2, alpha=0.8)
    ax2.annotate("fp16 reference 0.889", (0, 0.889), textcoords="offset points", xytext=(4, 6),
                 fontsize=8.5, color="#2166ac")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([a[0] for a in arc], fontsize=8.5)
    ax2.set_ylabel("macro-F1")
    ax2.set_title("Closing the int8-collapse arc")
    ax2.set_ylim(0.35, 0.95)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle("ReefScan-Edge — QAT int8 makes quantization free (DINOv2-B, A6000)", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"[qat-plot] wrote {OUT}  (control={'yes' if ctrl else 'no'})")


if __name__ == "__main__":
    main()
