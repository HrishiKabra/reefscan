"""DINOv2-B classifier + MAPIE-style split conformal. Phase 4/5.

Loads the trained head + backbone and the conformal calibration (qhat, LAC) from the HF
weights repo. Each SAM2 mask is classified on its bbox crop (resize 224 + ImageNet norm —
identical to training). Conformal LAC builds the prediction SET: include class k iff
1 - p_k <= qhat; never emit an empty set. Set size > 1 => uncertain => review_queue.

When weights are unavailable (stub mode), deterministic softmax is produced so the pipeline
and the uncertainty/review path are fully exercisable.
"""
from __future__ import annotations

import logging

from PIL import Image

from ..config import settings

logger = logging.getLogger(__name__)

CLASSES = list(settings.classes)  # ["healthy", "bleached"]
_STUB_VERSION = "reefscan-stub-v0"
_STUB_QHAT = 0.6  # threshold 0.4 -> segments within ~40-60 become uncertain (demo-visible)

_model = None
_tf = None
_qhat: float = _STUB_QHAT
_model_version: str = _STUB_VERSION


# ---------------------------------------------------------------------------
# Model (mirrors notebooks/01_train_dinov2 DINOv2Classifier)
# ---------------------------------------------------------------------------
def _build_model(num_classes: int):
    import torch.nn as nn
    from transformers import AutoModel

    class DINOv2Classifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
            self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

        def forward(self, x):
            o = self.backbone(pixel_values=x)
            cls = getattr(o, "pooler_output", None)
            if cls is None:
                cls = o.last_hidden_state[:, 0]
            return self.head(cls)

    return DINOv2Classifier()


def load() -> None:
    """Load DINOv2 head/backbone + conformal.json from the HF weights repo. Raises if absent."""
    global _model, _tf, _qhat, _model_version
    import json

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torchvision import transforms

    stage = settings.hf_stage
    w = hf_hub_download(settings.hf_repo, f"{stage}/model.safetensors", token=settings.hf_token)
    model = _build_model(len(CLASSES))
    model.load_state_dict(load_file(w))
    model.eval()
    _model = model

    _tf = transforms.Compose([
        transforms.Resize((settings.input_size, settings.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    cj = hf_hub_download(settings.hf_repo, f"{stage}/conformal.json", token=settings.hf_token)
    meta = json.load(open(cj))
    _qhat = float(meta["qhat"])
    _model_version = meta.get("model_version", f"reefscan-dinov2-coral-{stage}")
    logger.info("DINOv2 classifier loaded (version=%s, qhat=%.4f)", _model_version, _qhat)


def is_loaded() -> bool:
    return _model is not None


def qhat() -> float:
    return _qhat


def model_version() -> str:
    return _model_version


def classify(crop: Image.Image) -> dict[str, float]:
    """Return softmax {healthy, bleached} for a bbox crop."""
    if _model is None:
        return _stub_probs(crop)
    import torch

    x = _tf(crop.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        p = torch.softmax(_model(x), dim=1)[0].tolist()
    return {CLASSES[i]: float(p[i]) for i in range(len(CLASSES))}


def conformal_set(probs: dict[str, float], q: float) -> tuple[list[str], int]:
    """LAC split-conformal set: include class k iff (1 - p_k) <= qhat. Never empty."""
    keep = [c for c in CLASSES if (1.0 - probs[c]) <= q]
    if not keep:
        keep = [max(probs, key=probs.get)]
    # order by descending confidence for stable display
    keep.sort(key=lambda c: probs[c], reverse=True)
    return keep, len(keep)


def _stub_probs(crop: Image.Image) -> dict[str, float]:
    """Deterministic softmax from the crop's mean color — yields a mix incl. near-even
    (uncertain) segments so the review path is exercised."""
    import numpy as np

    arr = np.asarray(crop.convert("RGB").resize((16, 16)), dtype="float32") / 255.0
    # 'bluer / darker' -> leans bleached-ish; just a deterministic spread, not real signal
    h = float(np.clip(0.5 + (arr[..., 1].mean() - arr[..., 2].mean()) * 1.4, 0.05, 0.95))
    return {"healthy": h, "bleached": 1.0 - h}
