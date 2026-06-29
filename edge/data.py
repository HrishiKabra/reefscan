"""Data loaders for the edge harness — the SAME 1,565-image held-out test set (invariant #2)
and a representative train subsample for int8 calibration (invariant #3).

Pulls the dataset's parquet shards (fast) and preprocesses with the exact training transform
(resize 224 + ImageNet norm). Returns CPU tensors; the harness moves batches to the device.
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torchvision import transforms

DS = "NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset"
CLASSES = ("healthy", "bleached")
LABEL_MAP = {"CORAL": "healthy", "CORAL_BL": "bleached"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
_api = HfApi()


@lru_cache(maxsize=1)
def _basename_to_label() -> dict:
    m = {}
    for f in _api.list_repo_files(DS, repo_type="dataset"):
        p = f.split("/")
        if len(p) >= 3 and f.lower().endswith(".png"):
            m[p[-1]] = p[1]
    return m


def _shards(split: str) -> list[str]:
    name = {"train": "train", "val": "validation", "test": "test"}[split]
    return [f for f in _api.list_repo_files(DS, repo_type="dataset", revision="refs/convert/parquet")
            if f.endswith(".parquet") and f.split("/")[-2] == name]


def _items(split: str) -> list[tuple[bytes, int]]:
    import pyarrow.parquet as pq
    b2l = _basename_to_label()
    items = []
    for pf in _shards(split):
        path = hf_hub_download(DS, pf, repo_type="dataset", revision="refs/convert/parquet")
        for r in pq.read_table(path, columns=["image"]).column("image").to_pylist():
            c = LABEL_MAP.get(b2l.get(r["path"]))
            if c is not None:
                items.append((r["bytes"], CLASSES.index(c)))
    return items


def _to_tensor(items: list[tuple[bytes, int]]) -> tuple[torch.Tensor, np.ndarray]:
    xs = [TRANSFORM(Image.open(io.BytesIO(b)).convert("RGB")) for b, _ in items]
    return torch.stack(xs), np.array([y for _, y in items], dtype=np.int64)


def load_test() -> tuple[torch.Tensor, np.ndarray]:
    """The fixed 1,565-image test set: (images [N,3,224,224] float32 CPU, labels [N])."""
    return _to_tensor(_items("test"))


def load_calibration(n: int = 400) -> torch.Tensor:
    """Representative train subsample for int8 calibration (NOT random data — invariant #3).
    Strided sampling spans the full train set (both classes)."""
    items = _items("train")
    step = max(1, len(items) // n)
    sub = items[::step][:n]
    x, _ = _to_tensor(sub)
    return x
