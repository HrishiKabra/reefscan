"""Rung 4b — QAT int8. Closes the int8-collapse story (GPU-only; run on RunPod A6000 / Colab L4).

The int8 arc so far (edge/RESULTS.md):
  - naive ORT static int8 (MinMax, all ops) COLLAPSED to 0.399 F1 (majority class) — ViT activation
    outliers break per-tensor int8.
  - TensorRT PTQ with the IInt8EntropyCalibrator2 RECOVERED to 0.884 (TRT keeps LayerNorms in fp16),
    but on Ada int8 is dominated by fp16 (same speed, ~0.005 less F1).

This rung asks the natural next question: **is that residual int8 gap a calibration ceiling, or a
training one?** Quantization-Aware Training inserts fake-quant (Q/DQ) nodes and fine-tunes so the
network learns weights robust to int8 rounding — instead of quantizing a fixed fp model post-hoc.

Pipeline (NVIDIA TensorRT Model Optimizer = `modelopt`, the current QAT/PTQ toolkit for TRT):
  1. load the trained fp DINOv2-B (finetune stage, test F1 0.887),
  2. `mtq.quantize` with INT8 config + calibrate on a representative train subsample (QDQ inserted),
  3. record the PTQ-via-modelopt test F1 (the "before QAT" int8 number),
  4. QAT fine-tune a few epochs at low LR; keep the best-val checkpoint,
  5. export a **QDQ ONNX** and build a TRT int8 engine from it (TRT reads the embedded Q/DQ scales —
     explicit quantization, no calibrator),
  6. benchmark on the SAME 1,565-image test set via the harness -> a `tensorrt-int8-qat` row next to
     the PTQ `tensorrt-int8` row. Same invariants as harness.py.

Run: PYTHONPATH=. python3 edge/run_qat.py  (needs a CUDA GPU + `nvidia-modelopt[torch]` + tensorrt).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from edge.data import load_calibration, load_test, load_train, load_val
from edge.harness import append_results, benchmark
from edge.model import load_model
from edge.run_rung3 import ONNX_DIR
from edge.run_rung4 import PROFILE, make_predict

QAT_ONNX = os.path.join(ONNX_DIR, "dinov2_qat_qdq.onnx")
QAT_PLAN = os.path.join(ONNX_DIR, "dinov2_trt_int8_qat.plan")
CKPT = os.path.join(ONNX_DIR, "qat_best.pt")
HISTORY = "edge/docs/qat_history.json"

EPOCHS = int(os.environ.get("QAT_EPOCHS", "3"))
LR = float(os.environ.get("QAT_LR", "1e-5"))
BS = int(os.environ.get("QAT_BS", "64"))
CALIB_N = int(os.environ.get("QAT_CALIB_N", "512"))


@torch.no_grad()
def eval_f1(model, x, y, device, bs=64) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    preds = []
    for i in range(0, len(y), bs):
        preds.append(model(x[i:i + bs].to(device)).float().argmax(1).cpu().numpy())
    yp = np.concatenate(preds)
    return (float(f1_score(y, yp, labels=[0, 1], average="macro", zero_division=0)),
            float(accuracy_score(y, yp)))


def build_qdq_engine(onnx_path: str, plan_path: str) -> bytes:
    """Build a TRT int8 engine from a QDQ ONNX: explicit quantization (TRT uses the embedded Q/DQ
    scales), so NO calibrator — unlike the PTQ path in run_rung4. FP16 fallback for non-int8 layers."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[qat] ONNX parse error:", parser.get_error(i), flush=True)
            raise RuntimeError("TensorRT failed to parse the QDQ ONNX")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.INT8)   # honor the Q/DQ nodes (explicit precision)
    config.set_flag(trt.BuilderFlag.FP16)   # fp16 for whatever int8 can't take
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, *PROFILE)
    config.add_optimization_profile(profile)
    print("[qat] building int8 QAT engine from QDQ ONNX...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT int8-QAT engine build failed")
    data = bytes(serialized)   # IHostMemory -> bytes (has no len() itself)
    with open(plan_path, "wb") as f:
        f.write(data)
    print(f"[qat] saved {plan_path} ({len(data) / 1e6:.0f} MB)", flush=True)
    return data


def main():
    if not torch.cuda.is_available():
        raise SystemExit("[qat] QAT needs a CUDA GPU — run on RunPod A6000 / Colab L4.")
    import modelopt.torch.quantization as mtq
    device = "cuda"
    print(f"[qat] {torch.cuda.get_device_name(0)} | epochs={EPOCHS} lr={LR} bs={BS} calib_n={CALIB_N}",
          flush=True)
    os.makedirs(ONNX_DIR, exist_ok=True)   # checkpoints + QDQ ONNX + engine land here (before epoch 1 save)
    os.makedirs("edge/docs", exist_ok=True)

    model = load_model(stage="finetune", device=device)
    test_x, test_y = load_test()
    val_x, val_y = load_val()
    train_x, train_y = load_train()
    print(f"[qat] train={len(train_y)} val={len(val_y)} test={len(test_y)}", flush=True)

    f1_fp, acc_fp = eval_f1(model, test_x, test_y, device)
    print(f"[qat] baseline fp (finetune) test: F1={f1_fp:.4f} acc={acc_fp:.4f}", flush=True)

    # --- quantize + calibrate (QDQ inserted) ---
    calib = load_calibration(CALIB_N).to(device)

    def forward_loop(m):
        m.eval()
        with torch.no_grad():
            for i in range(0, len(calib), BS):
                m(calib[i:i + BS])

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)
    try:
        mtq.print_quant_summary(model)
    except Exception as e:  # summary is cosmetic
        print("[qat] (quant summary skipped:", e, ")", flush=True)

    f1_ptq, acc_ptq = eval_f1(model, test_x, test_y, device)
    print(f"[qat] modelopt PTQ (pre-QAT, fake-quant) test: F1={f1_ptq:.4f} acc={acc_ptq:.4f}", flush=True)

    # --- QAT fine-tune ---
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ty = torch.tensor(train_y, dtype=torch.long)
    idx = np.arange(len(train_y))
    history = [{"stage": "baseline_fp", "test_f1": round(f1_fp, 4)},
               {"stage": "modelopt_ptq", "test_f1": round(f1_ptq, 4)}]
    best_f1, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        np.random.shuffle(idx)
        tot, seen = 0.0, 0
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            xb = train_x[b].to(device)
            yb = ty[b].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); seen += len(b)
        vf1, vacc = eval_f1(model, val_x, val_y, device)
        tf1, tacc = eval_f1(model, test_x, test_y, device)
        print(f"[qat] epoch {ep + 1}/{EPOCHS}  train_loss={tot / seen:.4f}  "
              f"val_F1={vf1:.4f}  test_F1={tf1:.4f}", flush=True)
        history.append({"stage": f"qat_epoch_{ep + 1}", "train_loss": round(tot / seen, 4),
                        "val_f1": round(vf1, 4), "test_f1": round(tf1, 4)})
        if vf1 > best_f1:
            best_f1, best_state = vf1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, CKPT)
            print(f"[qat]   new best val_F1={vf1:.4f} -> {CKPT}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    f1_qat_torch, acc_qat_torch = eval_f1(model, test_x, test_y, device)
    print(f"[qat] best QAT (in-torch fake-quant) test: F1={f1_qat_torch:.4f} acc={acc_qat_torch:.4f}",
          flush=True)

    # --- export QDQ ONNX (do_constant_folding=False so Q/DQ nodes survive) ---
    model.eval()
    dummy = torch.randn(2, 3, 224, 224, device=device)
    os.makedirs(ONNX_DIR, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, dummy, QAT_ONNX,
                          input_names=["pixel_values"], output_names=["logits"],
                          dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, do_constant_folding=False)
    print(f"[qat] exported QDQ ONNX -> {QAT_ONNX} ({os.path.getsize(QAT_ONNX) / 1e6:.0f} MB)", flush=True)

    # --- build int8 engine from QDQ + benchmark on the harness ---
    import tensorrt as trt
    data = build_qdq_engine(QAT_ONNX, QAT_PLAN)
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(data)
    rows = benchmark("tensorrt-int8-qat", "tensorrt", "int8-qat", make_predict(engine),
                     test_x, test_y, "cuda")
    append_results(rows)
    for r in rows:
        print(f"[qat] {r['name']} batch={r['batch']:>2}  p50={r['p50_ms']:.2f}ms  "
              f"thrpt={r['throughput_ips']:.1f} img/s  macroF1={r['macro_f1']}  acc={r['accuracy']}",
              flush=True)

    f1_engine = rows[0]["macro_f1"]
    os.makedirs("edge/docs", exist_ok=True)
    history.append({"stage": "qat_engine_int8", "test_f1": f1_engine})
    json.dump({"epochs": EPOCHS, "lr": LR, "bs": BS, "calib_n": CALIB_N,
               "baseline_fp_f1": round(f1_fp, 4), "modelopt_ptq_f1": round(f1_ptq, 4),
               "qat_torch_f1": round(f1_qat_torch, 4), "qat_engine_int8_f1": f1_engine,
               "history": history}, open(HISTORY, "w"), indent=2)

    print("\n[qat] === int8 story ===", flush=True)
    print(f"[qat] fp16 (lossless)         : 0.8888", flush=True)
    print(f"[qat] ORT static int8 (naive) : 0.3992   (collapse)", flush=True)
    print(f"[qat] TRT PTQ entropy int8    : 0.8840   (recovers)", flush=True)
    print(f"[qat] TRT QAT int8 (this)     : {f1_engine}   "
          f"({'closes the gap to fp16' if f1_engine >= 0.887 else 'vs PTQ 0.884'})", flush=True)
    print(f"[qat] wrote {HISTORY}", flush=True)


if __name__ == "__main__":
    main()
