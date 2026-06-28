"""WaterNet underwater enhancement. Phase 5.

Currently an identity passthrough (real WaterNet weights are vendored later — the
WaterNet-over-CLAHE rationale stays a README talking point). Kept as a seam so the real
model drops in without touching the pipeline.
"""
from __future__ import annotations

from PIL import Image


def enhance(img: Image.Image) -> Image.Image:
    """Return an enhanced copy of the image. Identity for now."""
    return img
