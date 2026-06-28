"""Phase 1.5 follow-up: SAM2 AMG latency/quality sweep.

Sweeps points_per_side in {8, 16, 32} x image size in {384, 512} on CPU, reporting
AMG wall time and mask count for each. SAM2-Hiera-Small weights are already HF-cached.
"""
from __future__ import annotations

import io
import time

import numpy as np
import psutil
import torch
from PIL import Image

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)

POINTS = [8, 16, 32]
SIZES = [384, 512]


def get_coral_image() -> Image.Image:
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/Coral%20Outcrop%20Flynn%20Reef.jpg"
    try:
        import requests
        r = requests.get(url, timeout=30, headers={"User-Agent": "ReefScan-spike/0.1"})
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        print(f"[img] download failed ({e}); synthetic noise", flush=True)
        return Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))


def main() -> None:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    coral = get_coral_image()
    print(f"[img] source {coral.size}", flush=True)
    sam2_model = build_sam2_hf("facebook/sam2-hiera-small", device="cpu")

    results = []
    for size in SIZES:
        img_np = np.array(coral.resize((size, size)))
        for pps in POINTS:
            gen = SAM2AutomaticMaskGenerator(sam2_model, points_per_side=pps)
            t0 = time.perf_counter()
            masks = gen.generate(img_np)
            dt = time.perf_counter() - t0
            results.append((size, pps, dt, len(masks)))
            print(f"[run] size={size} pps={pps:<2} -> {dt:6.1f}s  {len(masks):>3} masks", flush=True)

    print("\n=== AMG SWEEP TABLE ===", flush=True)
    print(f"{'size':>5} | {'pps':>3} | {'prompts':>7} | {'time_s':>7} | {'masks':>5}", flush=True)
    print("-" * 42, flush=True)
    for size, pps, dt, nm in results:
        print(f"{size:>5} | {pps:>3} | {pps*pps:>7} | {dt:>7.1f} | {nm:>5}", flush=True)


if __name__ == "__main__":
    main()
