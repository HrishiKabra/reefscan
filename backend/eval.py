"""ReefScan evaluation harness. Phase 8 (portfolio hardening).

Loads the deployed DINOv2 model + its conformal calibration from the HF Hub, evaluates on
the NOAA test split, and writes to docs/eval/:
  - metrics.json            accuracy, per-class precision/recall/F1, ECE
  - confusion_matrix.png
  - reliability_diagram.png (calibration curve + ECE)
  - conformal.json          LAC vs APS: marginal + class-conditional coverage, avg set size
  - conformal_coverage.png

Calibrates conformal on the val split, evaluates coverage on test (proper split conformal).
Pure CPU; ~5-10 min for the full val+test (~3.1k images).

Run:  python -m backend.eval [--stage finetune|linear_probe] [--limit N]
"""
from __future__ import annotations

import argparse
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from transformers import AutoModel

REPO = "HrishiKabra/reefscan-dinov2-coral"
DS = "NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset"
CLASSES = ("healthy", "bleached")
LABEL_MAP = {"CORAL": "healthy", "CORAL_BL": "bleached"}
OUT = Path("docs/eval")
ALPHA = 0.10  # 90% target

_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# --------------------------------------------------------------------------- model
class DINOv2Classifier(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.head = nn.Linear(self.backbone.config.hidden_size, n)

    def forward(self, x):
        o = self.backbone(pixel_values=x)
        cls = getattr(o, "pooler_output", None)
        return self.head(cls if cls is not None else o.last_hidden_state[:, 0])


def load_model(stage: str) -> DINOv2Classifier:
    w = hf_hub_download(REPO, f"{stage}/model.safetensors")
    m = DINOv2Classifier(len(CLASSES))
    m.load_state_dict(load_file(w))
    return m.eval()


# --------------------------------------------------------------------------- data
def _shards(split: str) -> list[str]:
    name = {"train": "train", "val": "validation", "test": "test"}[split]
    files = HfApi().list_repo_files(DS, repo_type="dataset", revision="refs/convert/parquet")
    return [f for f in files if f.endswith(".parquet") and f.split("/")[-2] == name]


def _basename_to_label() -> dict:
    m = {}
    for f in HfApi().list_repo_files(DS, repo_type="dataset"):
        p = f.split("/")
        if len(p) >= 3 and f.lower().endswith(".png"):
            m[p[-1]] = p[1]
    return m


@torch.inference_mode()
def probs_for(model, split: str, b2l: dict, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    items = []
    for pf in _shards(split):
        path = hf_hub_download(DS, pf, repo_type="dataset", revision="refs/convert/parquet")
        for r in pq.read_table(path, columns=["image"]).column("image").to_pylist():
            cls = LABEL_MAP.get(b2l.get(r["path"]))
            if cls is not None:
                items.append((r["bytes"], CLASSES.index(cls)))
    if limit:
        items = items[:limit]
    P, Y, batch = [], [], []
    for i, (b, y) in enumerate(items):
        batch.append(_TF(Image.open(io.BytesIO(b)).convert("RGB")))
        Y.append(y)
        if len(batch) == 64 or i == len(items) - 1:
            P.append(torch.softmax(model(torch.stack(batch)), 1).numpy())
            batch = []
    return np.concatenate(P), np.array(Y)


# --------------------------------------------------------------------------- conformal
def lac_qhat(cal_p, cal_y):
    s = 1.0 - cal_p[np.arange(len(cal_y)), cal_y]
    n = len(s)
    return float(np.quantile(s, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0), method="higher"))


def lac_sets(p, qhat):
    sets = p >= (1.0 - qhat)
    empty = ~sets.any(1)
    sets[empty, p[empty].argmax(1)] = True
    return sets


def aps_scores(p, y):
    # APS calibration score: cumulative prob of classes ranked >= true class's prob
    order = np.argsort(-p, axis=1)
    ranks = np.argsort(order, axis=1)  # rank of each class
    sorted_p = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    true_rank = ranks[np.arange(len(y)), y]
    return cum[np.arange(len(y)), true_rank]


def aps_qhat(cal_p, cal_y):
    s = aps_scores(cal_p, cal_y)
    n = len(s)
    return float(np.quantile(s, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0), method="higher"))


def aps_sets(p, qhat):
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    keep_sorted = cum <= qhat
    keep_sorted[:, 0] = True  # always include top-1
    sets = np.zeros_like(p, dtype=bool)
    np.put_along_axis(sets, order, keep_sorted, axis=1)
    return sets


def coverage_report(sets, y):
    covered = sets[np.arange(len(y)), y]
    rep = {"marginal_coverage": float(covered.mean()),
           "avg_set_size": float(sets.sum(1).mean()),
           "class_conditional": {}}
    for ci, c in enumerate(CLASSES):
        mask = y == ci
        rep["class_conditional"][c] = {
            "coverage": float(covered[mask].mean()),
            "avg_set_size": float(sets[mask].sum(1).mean()),
            "n": int(mask.sum()),
        }
    return rep


# --------------------------------------------------------------------------- calibration (ECE)
def ece_and_curve(p, y, bins=10):
    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    xs, accs, confs, ece = [], [], [], 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        a, c = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(a - c)
        xs.append((edges[i] + edges[i + 1]) / 2); accs.append(a); confs.append(c)
    return float(ece), np.array(xs), np.array(accs), np.array(confs)


# --------------------------------------------------------------------------- plots
def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="GnBu")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_reliability(xs, accs, confs, ece, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    ax.plot(confs, accs, "o-", color="#1f9e89", label="model")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_coverage(lac, aps, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    groups = ["marginal", *CLASSES]
    lac_v = [lac["marginal_coverage"], *[lac["class_conditional"][c]["coverage"] for c in CLASSES]]
    aps_v = [aps["marginal_coverage"], *[aps["class_conditional"][c]["coverage"] for c in CLASSES]]
    x = np.arange(len(groups)); w = 0.35
    ax.bar(x - w / 2, lac_v, w, label="LAC", color="#1f9e89")
    ax.bar(x + w / 2, aps_v, w, label="APS", color="#f0a93b")
    ax.axhline(1 - ALPHA, ls="--", color="crimson", lw=1, label=f"target {1-ALPHA:.0%}")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("coverage"); ax.set_title("Conformal coverage: LAC vs APS")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="finetune", choices=["finetune", "linear_probe"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    print(f"[eval] loading {a.stage} model + data ...", flush=True)
    model = load_model(a.stage)
    b2l = _basename_to_label()
    cal_p, cal_y = probs_for(model, "val", b2l, a.limit)
    test_p, test_y = probs_for(model, "test", b2l, a.limit)
    print(f"[eval] val={len(cal_y)} test={len(test_y)}", flush=True)

    pred = test_p.argmax(1)
    report = classification_report(test_y, pred, labels=[0, 1], target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(test_y, pred, labels=[0, 1])
    ece, xs, accs, confs = ece_and_curve(test_p, test_y)

    lac_q = lac_qhat(cal_p, cal_y)
    aps_q = aps_qhat(cal_p, cal_y)
    lac = coverage_report(lac_sets(test_p.copy(), lac_q), test_y)
    aps = coverage_report(aps_sets(test_p, aps_q), test_y)

    plot_confusion(cm, OUT / "confusion_matrix.png")
    plot_reliability(xs, accs, confs, ece, OUT / "reliability_diagram.png")
    plot_coverage(lac, aps, OUT / "conformal_coverage.png")

    metrics = {
        "stage": a.stage, "n_test": int(len(test_y)), "n_cal": int(len(cal_y)),
        "accuracy": float((pred == test_y).mean()),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {c: report[c] for c in CLASSES},
        "ece": ece,
        "confusion_matrix": cm.tolist(),
    }
    conformal = {"alpha": ALPHA, "LAC": {"qhat": lac_q, **lac}, "APS": {"qhat": aps_q, **aps}}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "conformal.json").write_text(json.dumps(conformal, indent=2))

    print(f"\n[eval] acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} ECE={ece:.4f}")
    print(f"[eval] LAC cov={lac['marginal_coverage']:.4f} set={lac['avg_set_size']:.3f} | "
          f"APS cov={aps['marginal_coverage']:.4f} set={aps['avg_set_size']:.3f}")
    print("[eval] wrote docs/eval/{metrics.json,conformal.json,*.png}")


if __name__ == "__main__":
    main()
