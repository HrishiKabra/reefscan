"""EDA over the CoralNet annotations. Phase 2.

Its main job is to produce exactly what's needed to author backend/data/label_mapping.py
and choose a loss strategy. When run against real data it prints:

  1. Raw label frequency table  — every CoralNet label found, sorted by count, printed
     VERBATIM (no normalization/stripping) so the strings can be pasted into LABEL_MAPPING.
  2. Per-image label distribution — points/image and distinct-labels/image stats, plus a
     full image x label count matrix written to data/eda_per_image.csv.
  3. Class imbalance ratio — on the 4-class mapped distribution if LABEL_MAPPING is filled,
     otherwise on raw labels as a proxy.
  4. A recommendation — standard CE vs. class-weighting/oversampling vs. focal loss.

Run:  python -m backend.data.eda [--csv data/annotations.csv] [--per-image-out data/eda_per_image.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .label_mapping import CLASSES, LABEL_MAPPING, map_label


def _imbalance_report(counts: dict[str, int], basis: str) -> str:
    vals = [c for c in counts.values() if c > 0]
    if not vals:
        return f"imbalance ({basis}): no labeled samples."
    hi, lo = max(vals), min(vals)
    ratio = hi / lo
    lines = [f"imbalance ratio ({basis}) = {ratio:.1f}:1   (largest {hi} / smallest {lo})"]
    if ratio < 3:
        lines.append("  -> roughly balanced. Standard cross-entropy; no resampling needed.")
    elif ratio < 10:
        lines.append("  -> MODERATE imbalance. Use class-weighted cross-entropy "
                     "(weight = 1/freq) OR oversample minority classes.")
    else:
        lines.append("  -> SEVERE imbalance. Recommend focal loss (gamma~2) and/or "
                     "oversampling minority classes; consider undersampling the dominant class.")
    if lo < 50:
        lines.append(f"  !! smallest class has only {lo} samples — likely too few for a "
                     "reliable linear probe. Consider merging classes or collecting more.")
    return "\n".join(lines)


def run_eda(csv_path: str | Path, per_image_out: str | Path = "data/eda_per_image.csv") -> None:
    # dtype=str on label keeps the raw CoralNet strings EXACTLY as written (no coercion).
    df = pd.read_csv(csv_path, dtype={"label": str})
    n = len(df)
    n_images = df["image_name"].nunique()

    print(f"\n=== ReefScan EDA :: {csv_path} ===")
    print(f"annotations: {n}   images: {n_images}   "
          f"avg points/image: {n / max(n_images, 1):.1f}")

    # --- 1. Raw label frequency table (verbatim strings, sorted by count) -------------
    print("\n--- 1. RAW CoralNet label frequency (verbatim, sorted) ---")
    raw_counts = df["label"].value_counts()  # already sorted desc
    width = max((len(str(l)) for l in raw_counts.index), default=8)
    print(f"  {'label'.ljust(width)}  {'count':>7}  {'pct':>6}  -> mapped")
    for label, c in raw_counts.items():
        mapped = map_label(label)
        if mapped is not None:
            tag = mapped
        elif label in LABEL_MAPPING:
            tag = "DROP"
        else:
            tag = "UNMAPPED"
        print(f"  {str(label).ljust(width)}  {c:>7}  {100 * c / n:>5.1f}%  -> {tag}")

    unmapped = sorted(l for l in raw_counts.index if l not in LABEL_MAPPING)
    if unmapped:
        print(f"\n  !! {len(unmapped)} raw label(s) UNMAPPED — add each to LABEL_MAPPING:")
        print(f"     {unmapped}")

    # --- 2. Per-image label distribution ----------------------------------------------
    print("\n--- 2. Per-image label distribution ---")
    points_per_image = df.groupby("image_name").size()
    labels_per_image = df.groupby("image_name")["label"].nunique()
    print(f"  points/image   : min {points_per_image.min()}  median "
          f"{points_per_image.median():.0f}  mean {points_per_image.mean():.1f}  "
          f"max {points_per_image.max()}")
    print(f"  distinct labels/image: min {labels_per_image.min()}  median "
          f"{labels_per_image.median():.0f}  max {labels_per_image.max()}")
    matrix = pd.crosstab(df["image_name"], df["label"])  # image x label counts
    Path(per_image_out).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(per_image_out)
    print(f"  full image x label count matrix ({matrix.shape[0]} images x "
          f"{matrix.shape[1]} labels) written to: {per_image_out}")
    print("  head:")
    print(matrix.head(10).to_string().replace("\n", "\n    "))

    # --- 3 & 4. Imbalance ratio + recommendation --------------------------------------
    print("\n--- 3/4. Class imbalance + loss recommendation ---")
    if LABEL_MAPPING:
        mapped_counts = {cls: 0 for cls in CLASSES}
        for label, c in raw_counts.items():
            m = map_label(label)
            if m is not None:
                mapped_counts[m] += int(c)
        total_mapped = sum(mapped_counts.values()) or 1
        nclass = len(CLASSES)
        print(f"  {nclass}-class distribution:")
        for cls in CLASSES:
            c = mapped_counts[cls]
            print(f"    {cls:<16} {c:>7}  ({100 * c / total_mapped:.1f}%)")
        print(_imbalance_report(mapped_counts, basis=f"{nclass}-class"))
    else:
        print("  LABEL_MAPPING is EMPTY — reporting imbalance on RAW labels as a proxy.")
        print("  (Recompute after filling the mapping; the 4-class ratio is what matters.)")
        print(_imbalance_report(dict(raw_counts), basis="raw labels, PROXY"))
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/annotations.csv")
    ap.add_argument("--per-image-out", default="data/eda_per_image.csv")
    args = ap.parse_args()
    run_eda(args.csv, args.per_image_out)
