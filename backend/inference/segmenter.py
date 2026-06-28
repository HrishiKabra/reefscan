"""SAM2 segmentation. Phase 5.

SAM2-Hiera-Small via the Automatic Mask Generator (grid point prompts) — NOT manual
clicks, NOT a trained detection head (that is documented future work). Returns masks +
centroids; centroids feed classifier.py's patch crop.

AMG config is LOCKED by the Phase 1.5 sweep (read from config.settings, never inlined):
    points_per_side = 16 ; AMG input downscaled so longest edge = 512 px.

`segment()` returns a list of {bbox:[x0,y0,x1,y1], mask_area_px:int} in ORIGINAL image px.
When SAM2 isn't loaded (stub mode / weights absent), a deterministic grid of boxes is
returned so the whole pipeline + contract are exercisable without the heavy model.
"""
from __future__ import annotations

import logging

from PIL import Image

from ..config import settings

logger = logging.getLogger(__name__)

# Locked AMG config (Phase 1.5). Keep in sync with CLAUDE.md.
AMG_POINTS_PER_SIDE = settings.amg_points_per_side  # 16
AMG_INPUT_LONGEST_EDGE = settings.amg_longest_edge  # 512
# Phase 10 profiling: the AMG bottleneck is mask-decoding over the prompt grid (~92%), NOT
# the image encoder (~8%). points_per_batch=128 (vs the default 64) is a measured free ~8%
# speedup with identical masks; 256 regresses. See backend/optimize_sam2*.py + docs/eval/.
AMG_POINTS_PER_BATCH = 128

_amg = None  # loaded SAM2AutomaticMaskGenerator


def load() -> None:
    """Load SAM2-Hiera-Small + AMG once at startup. Raises if deps/weights unavailable."""
    global _amg
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
    from sam2.build_sam import build_sam2_hf  # type: ignore

    model = build_sam2_hf("facebook/sam2-hiera-small", device="cpu")
    _amg = SAM2AutomaticMaskGenerator(
        model, points_per_side=AMG_POINTS_PER_SIDE, points_per_batch=AMG_POINTS_PER_BATCH,
    )
    logger.info("SAM2 AMG loaded (pps=%d, ppb=%d)", AMG_POINTS_PER_SIDE, AMG_POINTS_PER_BATCH)


def is_loaded() -> bool:
    return _amg is not None


def _scale_to_longest(img: Image.Image, longest: int) -> tuple[Image.Image, float]:
    w, h = img.size
    scale = longest / max(w, h)
    if scale >= 1.0:
        return img, 1.0
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale)))), scale


def segment(img: Image.Image) -> list[dict]:
    if _amg is None:
        return _stub_segments(img)

    import numpy as np

    small, scale = _scale_to_longest(img, AMG_INPUT_LONGEST_EDGE)
    masks = _amg.generate(np.array(small))
    out: list[dict] = []
    for m in masks:
        x, y, w, h = m["bbox"]  # XYWH in downscaled coords
        out.append({
            "bbox": [round(x / scale), round(y / scale),
                     round((x + w) / scale), round((y + h) / scale)],
            "mask_area_px": int(m["area"] / (scale * scale)),
        })
    return out


def _stub_segments(img: Image.Image) -> list[dict]:
    """Deterministic 3x2 grid of boxes — contract-valid output without SAM2."""
    W, H = img.size
    cols, rows = 3, 2
    pad_x, pad_y = W * 0.04, H * 0.05
    cw, ch = (W - pad_x * (cols + 1)) / cols, (H - pad_y * (rows + 1)) / rows
    out: list[dict] = []
    sid = 0
    for r in range(rows):
        for c in range(cols):
            sid += 1
            x0 = pad_x * (c + 1) + cw * c
            y0 = pad_y * (r + 1) + ch * r
            # vary box size a little so coverage % differs per segment
            shrink = 0.82 + 0.12 * ((sid * 7) % 3) / 2
            bw, bh = cw * shrink, ch * shrink
            bbox = [round(x0), round(y0), round(x0 + bw), round(y0 + bh)]
            out.append({"bbox": bbox, "mask_area_px": round(bw * bh * 0.7)})
    return out
