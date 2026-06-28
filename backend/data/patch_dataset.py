"""Coral colony-patch classification dataset (ImageFolder format). Phase 2.

Built for NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset, which ships as an imagefolder:

    <root>/<split>/<RAW_LABEL>/<image>.PNG     e.g. train/CORAL/FFS-B013_2019_15_1024.PNG

These are pre-cropped, image-level colony patches (one health label per image) — NOT point
annotations. So training uses the WHOLE patch (resize -> 224, ImageNet-normalize); there is
no point/centroid crop here. Raw folder labels (CORAL, CORAL_BL) are collapsed to the
ReefScan classes via label_mapping.LABEL_MAPPING.

IMPORTANT: use the dataset's NATIVE train/val/test splits (the directory names). Do NOT
re-split with backend.data.split — the NOAA splits are site/year-controlled to prevent
leakage. backend.data.split is retained only for FUTURE point-annotated sources.

The inference-time bridge (Phase 5): SAM2 mask -> bbox crop -> resize 224 -> classify. The
training distribution (colony-level crops) closely matches those mask-bbox crops, so this
imagefolder design is a cleaner train/inference match than the original centroid-patch plan.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .label_mapping import CLASS_TO_IDX, LABEL_MAPPING, map_label
from .transforms import build_transform

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def scan_imagefolder(root: str | Path, split: str) -> list[tuple[Path, str]]:
    """Return [(image_path, raw_label), ...] for <root>/<split>/<RAW_LABEL>/*.

    raw_label is the immediate parent directory name (e.g. "CORAL", "CORAL_BL").
    """
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    samples: list[tuple[Path, str]] = []
    for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        raw_label = label_dir.name
        for img in label_dir.rglob("*"):
            if img.suffix.lower() in IMG_EXTENSIONS:
                samples.append((img, raw_label))
    return samples


class CoralPatchDataset(Dataset):
    """ImageFolder-style health-state classification dataset."""

    def __init__(
        self,
        samples: list[tuple[Path, str]],
        train: bool = False,
        allow_empty_mapping: bool = False,
    ) -> None:
        """
        Args:
            samples: list of (image_path, raw_label) — build with `from_imagefolder`.
            train: apply train-time augmentations if True.
            allow_empty_mapping: TEST-MODE ESCAPE HATCH. Must be True to construct while
                LABEL_MAPPING is empty. Real training/inference leaves this False so an
                unfilled mapping fails loudly instead of passing raw label strings as
                targets into Phase 3.
        """
        # Guard: never let an unfilled label_mapping.py reach training as a no-op.
        if not LABEL_MAPPING and not allow_empty_mapping:
            raise RuntimeError(
                "LABEL_MAPPING is empty — fill backend/data/label_mapping.py from the EDA "
                "raw-label table before building a real CoralPatchDataset. "
                "Pass allow_empty_mapping=True ONLY in tests/smoke checks."
            )

        self.transform = build_transform(train=train)

        kept: list[tuple[Path, int]] = []
        dropped: Counter = Counter()
        for path, raw in samples:
            mapped = map_label(raw)
            if mapped is None:
                dropped[raw] += 1
                continue
            kept.append((path, CLASS_TO_IDX[mapped]))
        if dropped:
            logger.warning("Dropping %d annotations with unmapped/None labels: %s",
                           sum(dropped.values()), dict(dropped))
        self.samples = kept
        if not kept:
            logger.warning(
                "CoralPatchDataset is EMPTY — LABEL_MAPPING covers none of the raw labels."
            )

    @classmethod
    def from_imagefolder(
        cls, root: str | Path, split: str, train: bool | None = None,
        allow_empty_mapping: bool = False,
    ) -> "CoralPatchDataset":
        """Build from <root>/<split>/<RAW_LABEL>/*. Augments iff split == 'train' (override
        with `train`)."""
        samples = scan_imagefolder(root, split)
        do_train = (split == "train") if train is None else train
        return cls(samples, train=do_train, allow_empty_mapping=allow_empty_mapping)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label_idx = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            tensor = self.transform(im)
        return tensor, label_idx
