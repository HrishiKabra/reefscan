"""Phase 1.5 feasibility spike.

Measures, on CPU, with REAL SAM2-Hiera-Small + REAL DINOv2-B loaded (WaterNet stubbed as
identity):
  1. peak process RAM with both models loaded simultaneously
  2. SAM2 Automatic Mask Generator time on one 512x512 image
  3. DINOv2-B forward time on one 224x224 patch

DINOv2 RAM + timing are measured BEFORE the (slow, CPU) SAM2 AMG generate, and every
number is flushed as it is produced, so partial results survive even if AMG is slow.

NOTE: this box is not HF Spaces. Numbers are indicative of the model footprint; the
Spaces budget interpretation (process + web server overhead) is applied in the writeup.
"""
from __future__ import annotations

import io
import sys
import time

import numpy as np
import psutil
import torch
from PIL import Image

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
_PROC = psutil.Process()


def rss_gb() -> float:
    return _PROC.memory_info().rss / (1024 ** 3)


def waternet_identity(img: Image.Image) -> Image.Image:
    """WaterNet stub — passthrough until real weights are vendored (Phase 5)."""
    return img


def get_coral_image(size: int = 512) -> Image.Image:
    """Download a Wikimedia coral image; fall back to synthetic noise if offline."""
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/Coral%20Outcrop%20Flynn%20Reef.jpg"
    try:
        import requests
        r = requests.get(url, timeout=30, headers={"User-Agent": "ReefScan-spike/0.1"})
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        print(f"[img] downloaded coral image {img.size}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[img] download failed ({e}); using synthetic noise", flush=True)
        img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
    return img.resize((size, size))


def main() -> None:
    base = rss_gb()
    print(f"[ram] baseline RSS: {base:.2f} GB", flush=True)

    coral = waternet_identity(get_coral_image(512))
    coral_np = np.array(coral)  # HxWx3 uint8 RGB

    # --- DINOv2-B ---
    from transformers import AutoImageProcessor, AutoModel
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    after_dino = rss_gb()
    print(f"[load] DINOv2-B loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[ram] after DINOv2-B: {after_dino:.2f} GB  (delta {after_dino - base:.2f})", flush=True)

    patch = coral.resize((224, 224))
    inputs = proc(images=patch, return_tensors="pt")
    with torch.inference_mode():
        for _ in range(2):  # warmup
            dino(**inputs)
        t0 = time.perf_counter()
        runs = 5
        for _ in range(runs):
            dino(**inputs)
        dino_ms = (time.perf_counter() - t0) / runs * 1000
    print(f"[time] DINOv2-B forward (224x224 patch): {dino_ms:.0f} ms/patch", flush=True)

    # --- SAM2-Hiera-Small ---
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf
    t0 = time.perf_counter()
    sam2_model = build_sam2_hf("facebook/sam2-hiera-small", device="cpu")
    peak = rss_gb()
    print(f"[load] SAM2-Hiera-Small loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[ram] PEAK with BOTH models loaded: {peak:.2f} GB  (delta vs baseline {peak - base:.2f})", flush=True)

    mask_gen = SAM2AutomaticMaskGenerator(sam2_model)
    print("[time] running SAM2 AMG on 512x512 (CPU, may be slow)...", flush=True)
    t0 = time.perf_counter()
    masks = mask_gen.generate(coral_np)
    amg_s = time.perf_counter() - t0
    print(f"[time] SAM2 AMG (512x512): {amg_s:.1f}s, {len(masks)} masks", flush=True)
    print(f"[ram] post-AMG RSS: {rss_gb():.2f} GB", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"baseline RAM         : {base:.2f} GB", flush=True)
    print(f"DINOv2-B only        : {after_dino:.2f} GB (+{after_dino - base:.2f})", flush=True)
    print(f"both models (PEAK)    : {peak:.2f} GB (+{peak - base:.2f})", flush=True)
    print(f"DINOv2-B forward      : {dino_ms:.0f} ms / 224 patch", flush=True)
    print(f"SAM2 AMG 512x512      : {amg_s:.1f} s / image, {len(masks)} masks", flush=True)


if __name__ == "__main__":
    sys.exit(main())
