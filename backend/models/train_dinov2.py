"""DINOv2-B health classifier — canonical training library. Phase 3.

This is the maintained, importable implementation. The Kaggle notebook
(notebooks/01_train_dinov2_kaggle.ipynb) mirrors this logic inline so it can run
self-contained on Kaggle; keep the two in sync.

Two stages (CLAUDE.md):
  - "linear_probe": freeze the DINOv2 backbone, train only the linear head (~30 min).
  - "finetune":     unfreeze the last 2 transformer blocks + final norm + head (~4 hr).

Hard Kaggle constraints honored:
  - checkpoint EVERY epoch to <work>/checkpoints/ (persisted as Kaggle output)
  - resume from the latest checkpoint if one exists (input dataset or working dir)
  - W&B logging throughout
  - 2-class head (healthy/bleached) driven by backend.data.CLASSES

Data: ImageFolder via backend.data.CoralPatchDataset using the NOAA NATIVE train/val/test
splits (do NOT re-split — see CLAUDE.md).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel

from backend.data import CLASSES, CoralPatchDataset

BACKBONE_NAME = "facebook/dinov2-base"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    stage: str = "linear_probe"            # "linear_probe" | "finetune"
    data_root: str = "/kaggle/input/noaa-coral-bleaching"  # imagefolder root (train/val/test)
    work_dir: str = "/kaggle/working"      # checkpoints written under <work_dir>/checkpoints
    resume_dir: str | None = None          # extra dir (e.g. prior run's output) to resume from
    backbone_name: str = BACKBONE_NAME
    epochs: int = 10                       # linear_probe ~10; finetune ~8 (fits 9h session)
    batch_size: int = 64
    num_workers: int = 4
    lr_head: float = 1e-3                  # linear_probe head lr; finetune head lr below
    lr_head_finetune: float = 1e-4
    lr_backbone: float = 1e-5              # finetune: last-2-block lr
    weight_decay: float = 0.05
    unfreeze_last_n: int = 2               # finetune: number of trailing transformer blocks
    label_smoothing: float = 0.0
    seed: int = 42
    # Hugging Face Hub
    hf_repo: str | None = None             # e.g. "user/reefscan-dinov2b" (provided before run)
    push_to_hub: bool = False
    # W&B
    wandb_project: str = "reefscan"
    wandb_run_name: str | None = None
    classes: list[str] = field(default_factory=lambda: list(CLASSES))

    @property
    def ckpt_dir(self) -> Path:
        return Path(self.work_dir) / "checkpoints" / self.stage


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DINOv2Classifier(nn.Module):
    """DINOv2 backbone -> CLS pooled embedding -> linear head."""

    def __init__(self, num_classes: int, backbone_name: str = BACKBONE_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        # DINOv2 returns pooler_output (CLS token); fall back to CLS of last_hidden_state.
        cls = getattr(out, "pooler_output", None)
        if cls is None:
            cls = out.last_hidden_state[:, 0]
        return self.head(cls)


def set_trainable(model: DINOv2Classifier, cfg: TrainConfig) -> list[dict]:
    """Freeze/unfreeze per stage; return optimizer param groups (with per-group lr)."""
    # Head is always trainable.
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    if cfg.stage == "linear_probe":
        return [{"params": model.head.parameters(), "lr": cfg.lr_head}]

    if cfg.stage == "finetune":
        # Unfreeze the last N transformer blocks + the final layernorm.
        blocks = model.backbone.encoder.layer
        for blk in blocks[-cfg.unfreeze_last_n:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(model.backbone, "layernorm"):
            for p in model.backbone.layernorm.parameters():
                p.requires_grad = True
        backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]
        return [
            {"params": model.head.parameters(), "lr": cfg.lr_head_finetune},
            {"params": backbone_trainable, "lr": cfg.lr_backbone},
        ]

    raise ValueError(f"unknown stage: {cfg.stage!r}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_dataloaders(cfg: TrainConfig) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        try:
            ds = CoralPatchDataset.from_imagefolder(cfg.data_root, split)
        except FileNotFoundError:
            continue  # test split optional
        loaders[split] = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=(split == "train"),
            num_workers=cfg.num_workers, pin_memory=True, drop_last=(split == "train"),
        )
    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError(f"need train+val imagefolders under {cfg.data_root}")
    return loaders


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(cfg: TrainConfig, model, optimizer, scheduler, scaler,
                    epoch: int, best_f1: float) -> Path:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save({
        "epoch": epoch,
        "best_f1": best_f1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "config": asdict(cfg),
        "classes": cfg.classes,
    }, path)
    # Also write/refresh a 'last.pt' pointer for easy resume.
    torch.save(torch.load(path, map_location="cpu"), cfg.ckpt_dir / "last.pt")
    return path


def _find_latest_checkpoint(cfg: TrainConfig) -> Path | None:
    candidates: list[Path] = []
    for base in [cfg.ckpt_dir, *( [Path(cfg.resume_dir)] if cfg.resume_dir else [] )]:
        if base and base.exists():
            last = base / "last.pt"
            if last.exists():
                candidates.append(last)
            candidates += sorted(base.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None


def maybe_resume(cfg: TrainConfig, model, optimizer, scheduler, scaler) -> tuple[int, float]:
    ckpt = _find_latest_checkpoint(cfg)
    if ckpt is None:
        return 0, -1.0
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    start_epoch = int(state["epoch"]) + 1
    best_f1 = float(state.get("best_f1", -1.0))
    print(f"[resume] from {ckpt} -> start_epoch={start_epoch}, best_f1={best_f1:.4f}")
    return start_epoch, best_f1


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    losses, ys, ps = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    return {"loss": float(np.mean(losses)), "acc": acc, "macro_f1": macro_f1, "report": report}


def run_training(cfg: TrainConfig, wandb=None) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders = get_dataloaders(cfg)
    model = DINOv2Classifier(len(cfg.classes), cfg.backbone_name).to(device)
    param_groups = set_trainable(model, cfg)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    start_epoch, best_f1 = maybe_resume(cfg, model, optimizer, scheduler, scaler)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] stage={cfg.stage} device={device} trainable_params={n_trainable:,} "
          f"epochs {start_epoch}->{cfg.epochs}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()
        train_loss = running / max(len(loaders["train"]), 1)
        val = evaluate(model, loaders["val"], device, criterion)
        best_f1 = max(best_f1, val["macro_f1"])
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
              f"val_macroF1={val['macro_f1']:.4f} (best {best_f1:.4f})")

        if wandb is not None:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                       "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
                       "lr": optimizer.param_groups[0]["lr"]})

        # CHECKPOINT EVERY EPOCH (Kaggle constraint).
        save_checkpoint(cfg, model, optimizer, scheduler, scaler, epoch, best_f1)

    # Final eval on test if present.
    results = {"best_val_macro_f1": best_f1}
    if "test" in loaders:
        results["test"] = evaluate(model, loaders["test"], device, criterion)
        print(f"[test] acc={results['test']['acc']:.4f} "
              f"macroF1={results['test']['macro_f1']:.4f}")

    _write_config(cfg, results)
    if cfg.push_to_hub and cfg.hf_repo:
        push_to_hub(cfg, model, results)
    return results


def _write_config(cfg: TrainConfig, results: dict) -> None:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.ckpt_dir / "config.json", "w") as f:
        json.dump({"config": asdict(cfg), "classes": cfg.classes, "results": _slim(results)},
                  f, indent=2)


def _slim(results: dict) -> dict:
    out = {k: v for k, v in results.items() if k != "test"}
    if "test" in results:
        out["test"] = {k: results["test"][k] for k in ("loss", "acc", "macro_f1")}
    return out


def push_to_hub(cfg: TrainConfig, model, results: dict) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(cfg.hf_repo, exist_ok=True)
    weights = cfg.ckpt_dir / "model.safetensors"
    try:
        from safetensors.torch import save_file
        save_file(model.state_dict(), str(weights))
    except Exception:
        weights = cfg.ckpt_dir / "model.pt"
        torch.save(model.state_dict(), weights)
    api.upload_file(path_or_fileobj=str(weights), path_in_repo=weights.name, repo_id=cfg.hf_repo)
    api.upload_file(path_or_fileobj=str(cfg.ckpt_dir / "config.json"),
                    path_in_repo=f"{cfg.stage}_config.json", repo_id=cfg.hf_repo)
    print(f"[hub] pushed weights + config to {cfg.hf_repo}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="linear_probe", choices=["linear_probe", "finetune"])
    ap.add_argument("--data-root", default="/kaggle/input/noaa-coral-bleaching")
    ap.add_argument("--work-dir", default="/kaggle/working")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--push-to-hub", action="store_true")
    a = ap.parse_args()
    run_training(TrainConfig(
        stage=a.stage, data_root=a.data_root, work_dir=a.work_dir, epochs=a.epochs,
        batch_size=a.batch_size, hf_repo=a.hf_repo, push_to_hub=a.push_to_hub,
    ))
