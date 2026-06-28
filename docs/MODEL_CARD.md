---
license: mit
tags:
  - image-classification
  - coral-reef
  - conformal-prediction
  - dinov2
  - ecology
datasets:
  - NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset
metrics:
  - accuracy
  - f1
pipeline_tag: image-classification
library_name: transformers
base_model: facebook/dinov2-base
---

# ReefScan — DINOv2-B coral health classifier (+ conformal calibration)

Binary coral-colony health classifier — **healthy** vs **bleached** — built on a
[DINOv2-B](https://huggingface.co/facebook/dinov2-base) backbone, with a packaged
**split-conformal calibration** so predictions can be returned as coverage-guaranteed
*prediction sets* rather than bare labels.

This repo backs **[ReefScan](https://github.com/HrishiKabra/reefscan)** — an end-to-end,
deployed reef-health pipeline (SAM2 segmentation → this classifier → conformal uncertainty →
observability + active-learning). Live demo: https://reefscan.vercel.app

## Files
```
linear_probe/  model.safetensors · config.json · conformal.json   # frozen backbone + linear head
finetune/      model.safetensors · config.json · conformal.json   # last 2 blocks + head unfrozen
```
Each `conformal.json` holds the LAC `qhat` (90% target coverage) calibrated on the held-out
val split. The deployed app serves **`finetune`**.

## Results (held-out test split, 1,565 images)

| stage | accuracy | macro-F1 | conformal coverage | avg. set size |
|---|---:|---:|---:|---:|
| linear probe | 0.857 | 0.846 | 0.914 | 1.120 |
| **fine-tune** | **0.895** | **0.887** | 0.923 | 1.075 |

**Per-class (fine-tune, test):** healthy — P 0.90 / R 0.93 / F1 0.92; bleached — P 0.88 / R 0.84 / F1 0.86.
**Calibration:** ECE = **0.046** (well-calibrated). **Confusion matrix:** `[[906, 68], [96, 495]]`.

**Conformal LAC vs APS** (90% target): LAC gives 0.923 marginal coverage at avg set size 1.075;
APS over-covers (0.996) with avg set 1.954. Class-conditional coverage under LAC is healthy 0.952
vs **bleached 0.876** — marginal conformal does not guarantee per-class coverage; the minority
class is under-covered (a known limitation; Mondrian/class-conditional conformal is the fix).

**Specialist vs. frontier VLM:** on the same test set, this specialist (0.895 acc / 0.887 F1 /
0.046 ECE) **beats zero-shot GPT-4o** (0.805 / 0.790 / 0.152) — ~9 pts accuracy and ~3× better
calibrated, at a fraction of the serving cost.

Coverage is the empirical fraction of test colonies whose conformal set contains the true
label (90% target; split conformal is designed to be slightly conservative). The fine-tune
is better-calibrated: average set size drops to 1.075, i.e. ~92% of colonies get a single
confident label and only ~8% are flagged uncertain for human review.

## Intended use
Decision-support for reef-health screening from underwater imagery: flag likely-bleached
colonies and surface uncertain cases for expert review. **Not** a substitute for in-water
expert assessment. Trained on NOAA-PIFSC Pacific reef imagery; performance on other regions,
gear, or water conditions is unverified.

## Training data
[NMFS-OSI/NOAA-PIFSC-ESD Coral Bleaching Dataset](https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset)
— ~10,419 pre-cropped colony patches, image-level labels (`CORAL`=healthy, `CORAL_BL`=bleached),
native site/year-controlled train/val/test splits, ~62% healthy / 38% bleached.

## Training procedure
DINOv2-B backbone, 224×224 inputs, ImageNet normalization, AdamW, cosine schedule, 10 epochs,
cross-entropy. **Linear probe** freezes the backbone (1,538-param head); **fine-tune** additionally
unfreezes the last 2 transformer blocks + final norm. Trained on a single free GPU session
(Kaggle/Colab); checkpointed every epoch and resumable. See the
[training notebook](https://github.com/HrishiKabra/reefscan/blob/main/notebooks/01_train_dinov2.ipynb).

## Conformal calibration
Split-conformal **LAC** (Least Ambiguous set-valued Classifier): nonconformity score
`s = 1 − p[true]`, `qhat` = the `⌈(n+1)(1−α)⌉/n` quantile on the val split (α = 0.10). A test
colony's set is `{ k : p_k ≥ 1 − qhat }` (never empty). Set size > 1 ⇒ uncertain ⇒ routed to
human review in the app. This is what turns a point classifier into a calibrated,
review-triggering one.

## Usage
```python
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel
import torch.nn as nn

STAGE = "finetune"
class DINOv2Classifier(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.head = nn.Linear(self.backbone.config.hidden_size, n)
    def forward(self, x):
        o = self.backbone(pixel_values=x)
        return self.head(o.pooler_output)

m = DINOv2Classifier()
m.load_state_dict(load_file(hf_hub_download("HrishiKabra/reefscan-dinov2-coral", f"{STAGE}/model.safetensors")))
m.eval()
qhat = json.load(open(hf_hub_download("HrishiKabra/reefscan-dinov2-coral", f"{STAGE}/conformal.json")))["qhat"]
# probs -> conformal set: classes = ["healthy","bleached"]; keep k where p_k >= 1 - qhat
```

## Limitations & bias
Binary only (`dead`/`algae_covered` are future work); single-region training data; whole-patch
classification (no fine localization within a colony); WaterNet enhancement is currently a
passthrough. The conformal guarantee is *marginal* over the test distribution — it holds in
aggregate, not per-individual-image, and assumes the deployment distribution matches calibration.

## Citation
Dataset: NMFS-OSI / NOAA-PIFSC-ESD Coral Bleaching Dataset. Backbone: DINOv2 (Meta AI).
Project: https://github.com/HrishiKabra/reefscan
