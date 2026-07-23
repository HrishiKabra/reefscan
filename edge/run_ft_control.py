"""Control for run_qat.py — isolate 'extra fine-tuning' from 'quantization recovery'.

QAT int8 reached test macro-F1 0.900, above the fp checkpoint's 0.885. But QAT is ALSO 3 epochs of
extra fine-tuning, so part of that lift is just training. This control answers the question cleanly:
fine-tune the SAME fp model for the SAME 3 epochs (same lr/bs/data/best-val selection) with NO
quantization, and measure its test F1.

  - if fp-finetuned ~= 0.900  -> the lift is training; QAT int8 MATCHES equally-trained fp16
                                 => "int8 is free" (no accuracy cost vs a fair fp16 baseline).
  - if fp-finetuned  < 0.900  -> QAT's fake-quant noise also regularized; int8-QAT beats fp16.

Reuses run_qat.eval_f1 + the identical loop so the only difference is the absence of mtq.quantize.
Run: PYTHONPATH=. python3 edge/run_ft_control.py   (CUDA GPU; no modelopt needed).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from edge.data import load_test, load_train, load_val
from edge.model import load_model
from edge.run_qat import EPOCHS, LR, BS, eval_f1

OUT = "edge/docs/qat_control.json"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("[ctrl] needs a CUDA GPU.")
    device = "cuda"
    print(f"[ctrl] {torch.cuda.get_device_name(0)} | fp fine-tune control | epochs={EPOCHS} lr={LR} bs={BS}",
          flush=True)

    model = load_model(stage="finetune", device=device)
    test_x, test_y = load_test()
    val_x, val_y = load_val()
    train_x, train_y = load_train()

    f1_fp, acc_fp = eval_f1(model, test_x, test_y, device)
    print(f"[ctrl] baseline fp (no finetune) test: F1={f1_fp:.4f} acc={acc_fp:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ty = torch.tensor(train_y, dtype=torch.long)
    idx = np.arange(len(train_y))
    history = [{"stage": "baseline_fp", "test_f1": round(f1_fp, 4)}]
    best_f1, best_test = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        np.random.shuffle(idx)
        tot, seen = 0.0, 0
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            logits = model(train_x[b].to(device))
            loss = F.cross_entropy(logits, ty[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); seen += len(b)
        vf1, _ = eval_f1(model, val_x, val_y, device)
        tf1, tacc = eval_f1(model, test_x, test_y, device)
        print(f"[ctrl] epoch {ep + 1}/{EPOCHS}  train_loss={tot / seen:.4f}  val_F1={vf1:.4f}  test_F1={tf1:.4f}",
              flush=True)
        history.append({"stage": f"ft_epoch_{ep + 1}", "train_loss": round(tot / seen, 4),
                        "val_f1": round(vf1, 4), "test_f1": round(tf1, 4)})
        if vf1 > best_f1:
            best_f1, best_test = vf1, tf1  # best-val selection, same as QAT

    os.makedirs("edge/docs", exist_ok=True)
    json.dump({"epochs": EPOCHS, "lr": LR, "bs": BS,
               "baseline_fp_f1": round(f1_fp, 4),
               "ft_control_best_test_f1": round(best_test, 4),
               "qat_int8_f1": 0.8996, "note": "best-val checkpoint, same protocol as run_qat.py",
               "history": history}, open(OUT, "w"), indent=2)
    print(f"\n[ctrl] === control result ===", flush=True)
    print(f"[ctrl] fp (no finetune)      : {f1_fp:.4f}", flush=True)
    print(f"[ctrl] fp + {EPOCHS}ep finetune   : {best_test:.4f}  (control)", flush=True)
    print(f"[ctrl] QAT int8 ({EPOCHS}ep)       : 0.8996", flush=True)
    gap = 0.8996 - best_test
    print(f"[ctrl] int8-QAT minus fp-finetuned = {gap:+.4f}  "
          f"({'int8 is free vs equally-trained fp16' if abs(gap) < 0.004 else ('int8-QAT still ahead' if gap > 0 else 'fp-finetune ahead')})",
          flush=True)
    print(f"[ctrl] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
