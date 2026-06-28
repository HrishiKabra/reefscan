"""Follow-up: the AMG bottleneck is mask-decoding over the prompt grid (~92%), not the
encoder. This tests the actionable lever there — `points_per_batch` (how many grid prompts
the mask decoder processes per forward) — plus a lighter grid, holding quality in view.
Writes docs/eval/sam2_opt_decode.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path("docs/eval")


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    model = build_sam2_hf("facebook/sam2-hiera-small", device="cpu")
    img = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)

    configs = [
        ("pps16_ppb64 (baseline)", dict(points_per_side=16, points_per_batch=64)),
        ("pps16_ppb128", dict(points_per_side=16, points_per_batch=128)),
        ("pps16_ppb256", dict(points_per_side=16, points_per_batch=256)),
        ("pps12_ppb128", dict(points_per_side=12, points_per_batch=128)),
    ]
    rows = []
    for name, kw in configs:
        gen = SAM2AutomaticMaskGenerator(model, **kw)
        gen.generate(img)  # warmup
        t = time.perf_counter()
        masks = gen.generate(img)
        dt = round((time.perf_counter() - t) * 1000)
        rows.append({"config": name, "ms": dt, "n_masks": len(masks),
                     "points": kw["points_per_side"] ** 2, **kw})
        print(f"[decode] {name:26s} -> {dt:6d} ms, {len(masks):3d} masks", flush=True)

    base = rows[0]["ms"]
    for r in rows:
        r["speedup_vs_baseline"] = round(base / r["ms"], 2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sam2_opt_decode.json").write_text(json.dumps(rows, indent=2))
    print("\n" + json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
