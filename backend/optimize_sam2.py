"""Explore optimizing the SAM2 image encoder — the profiled latency bottleneck. Phase 10.

First it splits the AMG wall-clock into (a) the image encoder (one `set_image`) and (b) the
per-point mask decoding for the 16x16 prompt grid — because that ratio decides whether
optimizing the *encoder* is even the right lever. Then it tries two CPU-friendly encoder
optimizations and measures each: bf16 autocast and torch.compile. Writes docs/eval/sam2_opt.json.

Run:  python -m backend.optimize_sam2
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path("docs/eval")


def med_ms(fn, n=3, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return round(sorted(ts)[len(ts) // 2] * 1000)


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("[opt] loading SAM2-Hiera-Small ...", flush=True)
    model = build_sam2_hf("facebook/sam2-hiera-small", device="cpu")
    img = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    pred = SAM2ImagePredictor(model)
    res: dict = {}

    # ---- 1. encoder (set_image) vs full AMG ----
    print("[opt] timing image encoder (set_image) ...", flush=True)
    enc = med_ms(lambda: pred.set_image(img), n=3, warm=1)
    print("[opt] timing full AMG (pps=16 @ 512) ...", flush=True)
    amg = SAM2AutomaticMaskGenerator(model, points_per_side=16)
    full = med_ms(lambda: amg.generate(img), n=1, warm=0)
    res["encoder_ms"] = enc
    res["full_amg_ms"] = full
    res["mask_decode_ms"] = max(full - enc, 0)
    res["encoder_share_pct"] = round(enc / full * 100, 1) if full else None

    # ---- 2. bf16 autocast on the encoder ----
    print("[opt] bf16 autocast ...", flush=True)
    try:
        def enc_bf16():
            with torch.autocast("cpu", dtype=torch.bfloat16):
                pred.set_image(img)
        b = med_ms(enc_bf16, n=3, warm=1)
        res["encoder_bf16_ms"] = b
        res["bf16_speedup"] = round(enc / b, 2)
    except Exception as e:  # noqa: BLE001
        res["bf16_note"] = f"{type(e).__name__}: {str(e)[:100]}"

    # ---- 3. torch.compile the encoder ----
    print("[opt] torch.compile (may take a minute to compile) ...", flush=True)
    try:
        model.image_encoder = torch.compile(model.image_encoder)
        c = med_ms(lambda: pred.set_image(img), n=3, warm=2)  # warmups trigger compilation
        res["encoder_compiled_ms"] = c
        res["compile_speedup"] = round(enc / c, 2)
    except Exception as e:  # noqa: BLE001
        res["compile_note"] = f"{type(e).__name__}: {str(e)[:100]}"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sam2_opt.json").write_text(json.dumps(res, indent=2))
    print("\n[opt] RESULT\n" + json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
