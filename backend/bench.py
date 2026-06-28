"""Inference profiling + optimization benchmark. Phase 8.

(1) Profiles the pipeline stages (SAM2 AMG vs DINOv2 forward) to *prove* where the latency
    goes, and (2) demonstrates an optimization on the classifier: fp32 vs CPU dynamic-int8
    quantization (and ONNX Runtime if exportable), with measured speedups.

Conclusion the numbers support: SAM2's image encoder dominates wall-clock; it's the real
ONNX/quantization target. DINOv2 is already cheap, but quantizing it is a clean,
measurable demonstration of the technique. Writes docs/eval/bench.json.

Run:  python -m backend.bench [--amg]   (--amg also times SAM2, which is slow on CPU)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

OUT = Path("docs/eval")
_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class DINOv2Classifier(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.head = nn.Linear(self.backbone.config.hidden_size, n)

    def forward(self, x):
        o = self.backbone(pixel_values=x)
        cls = getattr(o, "pooler_output", None)
        return self.head(cls if cls is not None else o.last_hidden_state[:, 0])


def _time(fn, n=10, warmup=2):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000  # ms


def bench_classifier() -> dict:
    # select an available CPU quantized engine (qnnpack on ARM, fbgemm on x86)
    for eng in ("qnnpack", "fbgemm"):
        if eng in getattr(torch.backends.quantized, "supported_engines", []):
            torch.backends.quantized.engine = eng
            break

    model = DINOv2Classifier().eval()
    x = torch.randn(1, 3, 224, 224)
    res = {"quant_engine": torch.backends.quantized.engine}
    with torch.inference_mode():
        res["fp32_ms"] = round(_time(lambda: model(x)), 1)
        try:
            qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            res["int8_dynamic_ms"] = round(_time(lambda: qmodel(x)), 1)
            res["speedup_int8"] = round(res["fp32_ms"] / res["int8_dynamic_ms"], 2)
        except Exception as e:  # noqa: BLE001
            res["int8_note"] = f"dynamic quant unavailable: {type(e).__name__}: {str(e)[:80]}"

    # Optional ONNX Runtime path (best-effort; export can be finicky for ViTs)
    try:
        import onnxruntime as ort  # type: ignore
        onnx_path = OUT / "dinov2_classifier.onnx"
        OUT.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, x, str(onnx_path), opset_version=17,
                          input_names=["pixel_values"], output_names=["logits"],
                          dynamic_axes={"pixel_values": {0: "b"}, "logits": {0: "b"}})
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        xn = x.numpy()
        res["onnx_fp32_ms"] = round(_time(lambda: sess.run(None, {"pixel_values": xn})), 1)
        res["speedup_onnx"] = round(res["fp32_ms"] / res["onnx_fp32_ms"], 2)
    except Exception as e:  # noqa: BLE001
        res["onnx_note"] = f"ONNX path skipped ({type(e).__name__}); install onnx+onnxruntime to enable"
    return res


def bench_sam2() -> dict:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    from .config import settings
    sam = build_sam2_hf("facebook/sam2-hiera-small", device="cpu")
    gen = SAM2AutomaticMaskGenerator(sam, points_per_side=settings.amg_points_per_side)
    img = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    t = time.perf_counter()
    masks = gen.generate(img)
    return {"amg_ms": round((time.perf_counter() - t) * 1000, 0), "n_masks": len(masks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amg", action="store_true", help="also time SAM2 AMG (slow on CPU)")
    a = ap.parse_args()
    torch.set_num_threads(__import__("os").cpu_count() or 4)
    OUT.mkdir(parents=True, exist_ok=True)

    print("[bench] classifier (DINOv2-B) ...", flush=True)
    clf = bench_classifier()
    print(json.dumps(clf, indent=2))

    out = {"classifier": clf}
    if a.amg:
        print("[bench] SAM2 AMG (pps=16 @ 512) ...", flush=True)
        sam = bench_sam2()
        print(json.dumps(sam, indent=2))
        out["sam2"] = sam
        # the headline: SAM2 share of a ~27-mask image
        per_img_dino = clf["fp32_ms"] * sam["n_masks"]
        out["bottleneck"] = {
            "sam2_ms": sam["amg_ms"],
            "dino_total_ms": round(per_img_dino),
            "sam2_share_pct": round(sam["amg_ms"] / (sam["amg_ms"] + per_img_dino) * 100, 1),
        }
        print("\n[bench] SAM2 is", out["bottleneck"]["sam2_share_pct"], "% of a full image's compute")

    (OUT / "bench.json").write_text(json.dumps(out, indent=2))
    print("[bench] wrote docs/eval/bench.json")


if __name__ == "__main__":
    main()
