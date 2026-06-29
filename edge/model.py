"""DINOv2-B classifier — same architecture as training; loads the trained checkpoint.

Reuses the deployed weights on the HF Hub (HrishiKabra/reefscan-dinov2-coral, finetune stage).
Self-contained so the edge harness is portable to Colab without the backend package.
"""
from __future__ import annotations

import torch.nn as nn

REPO = "HrishiKabra/reefscan-dinov2-coral"
STAGE = "finetune"
BACKBONE = "facebook/dinov2-base"
CLASSES = ("healthy", "bleached")


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = BACKBONE):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(backbone)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        o = self.backbone(pixel_values=x)
        cls = getattr(o, "pooler_output", None)
        return self.head(cls if cls is not None else o.last_hidden_state[:, 0])


def load_model(repo: str = REPO, stage: str = STAGE, device: str = "cpu") -> DINOv2Classifier:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    w = hf_hub_download(repo, f"{stage}/model.safetensors")
    m = DINOv2Classifier(len(CLASSES))
    m.load_state_dict(load_file(w))
    m.eval()
    return m.to(device)
