"""Train/val/test split, stratified BY IMAGE on the image's majority class. Phase 2.

RESERVED FOR FUTURE POINT-ANNOTATED SOURCES — NOT used for the current NOAA dataset.
The NOAA-PIFSC bleaching dataset ships with native, site/year-controlled train/val/test
splits, so we use those directly (see patch_dataset.CoralPatchDataset.from_imagefolder)
and must NOT re-split it here (that would risk the very site leakage NOAA already
controlled for). This module is kept for a future CoralNet-style point-annotated source.

Splitting by image (not by point) prevents leakage — many points share one image, so a
point-level split would put correlated patches from the same image in both train and val.
Each image is assigned its majority mapped label and stratified on that.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .label_mapping import map_label

DEFAULT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def _split_once(frame: pd.DataFrame, test_size: float, random_state: int,
                stratify_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train_test_split, stratified when possible, falling back to random.

    Stratification needs >=2 samples per class AND each resulting partition to hold at
    least one sample per class; on tiny/dev data that can't hold, so we fall back rather
    than crash. On real data (thousands of images) the stratified path is taken.
    """
    strat = frame[stratify_col]
    if strat.value_counts().min() >= 2:
        try:
            return train_test_split(frame, test_size=test_size,
                                    random_state=random_state, stratify=strat)
        except ValueError:
            pass  # partition too small to carry every class -> fall back
    return train_test_split(frame, test_size=test_size, random_state=random_state)


def _image_majority_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One row per image with its majority mapped label (unmapped points ignored)."""
    d = df.copy()
    d["mapped"] = d["label"].map(map_label)
    d = d[d["mapped"].notna()]
    if len(d) == 0:
        return pd.DataFrame(columns=["image_name", "majority"])
    majority = (
        d.groupby("image_name")["mapped"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"mapped": "majority"})
    )
    return majority


def stratified_split_by_image(
    annotations_csv: str | Path,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return {'train','val','test'} -> annotation DataFrames (point rows).

    Stratification falls back to a non-stratified split for any class too small to
    appear in every split (common with tiny/dev data).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    df = pd.read_csv(annotations_csv)
    img_labels = _image_majority_labels(df)
    if len(img_labels) < 3:
        raise ValueError(
            f"Need >=3 labeled images to split; got {len(img_labels)}. "
            "This is expected with the synthetic stub data — provide real data."
        )

    train_r, val_r, test_r = ratios
    train_imgs, hold_imgs = _split_once(
        img_labels, test_size=(val_r + test_r),
        random_state=random_state, stratify_col="majority",
    )
    rel_test = test_r / (val_r + test_r)
    val_imgs, test_imgs = _split_once(
        hold_imgs, test_size=rel_test,
        random_state=random_state, stratify_col="majority",
    )

    def rows_for(imgs: pd.DataFrame) -> pd.DataFrame:
        return df[df["image_name"].isin(imgs["image_name"])].reset_index(drop=True)

    return {"train": rows_for(train_imgs), "val": rows_for(val_imgs), "test": rows_for(test_imgs)}
