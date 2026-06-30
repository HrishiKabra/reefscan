"""Rung 3b — precision: fp16 + int8 PTQ, plus a TF32 control. The accuracy-vs-latency core.

Registers three variants:
  1. pytorch-tf32  (control) — eager PyTorch fp32 with TF32 matmuls ENABLED. Isolates how much of
     the ONNX-Runtime win was graph fusion vs TF32 tensor cores (PyTorch's eager default is TF32 off).
  2. onnxruntime-fp16 — the ONNX graph cast to fp16 (keep_io_types=True, so it still takes/returns
     fp32; only the compute is fp16 -> full tensor-core throughput). The headline precision win.
  3. onnxruntime-int8 — static PTQ (QDQ, per-channel), calibrated on the REPRESENTATIVE train
     subsample (data.load_calibration() — invariant #3, NOT random). This is the DOCUMENTED NEGATIVE:
     static int8 PTQ COLLAPSES this DINOv2-B to ~majority-class (verified ~0.43-0.48 macro-F1 across
     naive vs MatMul-only quantization x MinMax vs Entropy calibration). ViTs have heavy-tailed
     activation outliers that static int8 calibration squashes. The viable int8 path is QAT or
     TensorRT's int8 (entropy calibrator + int8 tensor-core kernels) — Rung 4. Latency on CPU
     because ORT's CUDA EP doesn't accelerate int8 at all.

Everything is CPU-runnable for local verification (fp16/int8 accuracy is the same anywhere); the
GPU latency rows (tf32, fp16) come from Colab. Run AFTER run_rung3 (reuses its fp32 ONNX export).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from edge.data import load_calibration, load_test
from edge.harness import append_results, benchmark
from edge.model import load_model
from edge.run_rung3 import ONNX_DIR, ONNX_PATH, export_onnx, make_predict, make_session

FP16_PATH = os.path.join(ONNX_DIR, "dinov2_fp16.onnx")
PREP_PATH = os.path.join(ONNX_DIR, "dinov2_fp32_prep.onnx")
INT8_PATH = os.path.join(ONNX_DIR, "dinov2_int8.onnx")
INPUT_NAME = "pixel_values"


def export_fp16() -> None:
    # NOTE: onnxconverter-common 1.14 uses np.fromstring/ndarray.tostring (both removed in numpy>=2),
    # so this needs numpy<2 — satisfied by the pinned numpy 1.26.4 (also avoids numpy-2 ABI issues
    # across torch / onnxruntime wheels). keep_io_types=True -> fp32 in/out, fp16 compute internally.
    import onnx
    from onnxconverter_common import float16
    m16 = float16.convert_float_to_float16(onnx.load(ONNX_PATH), keep_io_types=True)
    onnx.save(m16, FP16_PATH)
    print(f"[rung3b] fp16 ONNX -> {FP16_PATH} ({os.path.getsize(FP16_PATH) / 1e6:.0f} MB)", flush=True)


def export_int8(calib: torch.Tensor, out_path: str, op_types=None) -> None:
    """Static int8 PTQ (QDQ, per-channel), calibrated on `calib`. `op_types` restricts which ops
    get quantized; None = all (naive — collapses transformers), ["MatMul"] = the heavy GEMMs only,
    leaving LayerNorm/GELU/softmax in fp32 (the correct, accuracy-preserving transformer recipe)."""
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if not os.path.exists(PREP_PATH):
        quant_pre_process(ONNX_PATH, PREP_PATH, skip_symbolic_shape=False)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter([{INPUT_NAME: calib[i:i + 1].numpy().astype(np.float32)}
                             for i in range(calib.shape[0])])

        def get_next(self):
            return next(self._it, None)

    kw = {"op_types_to_quantize": op_types} if op_types else {}
    quantize_static(PREP_PATH, out_path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8, **kw)
    print(f"[rung3b] int8 ONNX -> {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB) "
          f"(ops={op_types or 'ALL'}, calibrated on {calib.shape[0]} representative train images)",
          flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rung3b] device={device}", flush=True)
    model = load_model(device=device)
    test_x, test_y = load_test()
    if not os.path.exists(ONNX_PATH):
        export_onnx(model, device)

    rows = []

    # 1. TF32 control (GPU only — TF32 is an Ampere+ tensor-core path; on CPU it's a no-op label).
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[rung3b] TF32 enabled — benchmarking pytorch-tf32 control...", flush=True)
        rows += benchmark("pytorch-tf32", "pytorch", "tf32", lambda xb: model(xb),
                          test_x, test_y, "cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # 2. ONNX Runtime fp16 (compute in fp16, fp32 IO).
    export_fp16()
    sess16 = make_session(device, FP16_PATH)
    rows += benchmark("onnxruntime-fp16", "onnxruntime", "fp16", make_predict(sess16, device),
                      test_x, test_y, device)

    # 3. ONNX Runtime int8 static PTQ — the DOCUMENTED NEGATIVE. Static int8 PTQ collapses this ViT
    #    (verified: ~0.43-0.48 macro-F1 ≈ majority-class, across naive/MatMul-only x MinMax/Entropy
    #    calibration). DINOv2's activation outliers break static int8; the viable int8 path is QAT or
    #    TensorRT's specialized int8 (entropy calibrator + kernels) — Rung 4. We still record the row so
    #    the collapse is a measured artifact, not a claim. Latency on CPU (ORT CUDA EP has no int8).
    export_int8(load_calibration(), INT8_PATH)
    sess8 = make_session("cpu", INT8_PATH)
    rows += benchmark("onnxruntime-int8", "onnxruntime", "int8", make_predict(sess8, "cpu"),
                      test_x, test_y, "cpu")
    f1_int8 = next(r["macro_f1"] for r in rows if r["name"] == "onnxruntime-int8")
    if f1_int8 < 0.7:
        print(f"[rung3b] NOTE: int8 static PTQ collapsed (macro-F1={f1_int8}) — expected; see Rung 4 "
              f"(TensorRT entropy-calibrated int8) for viable GPU int8.", flush=True)

    append_results(rows)
    for r in rows:
        print(f"[rung3b] {r['name']:<18} dev={r['device']:<4} batch={r['batch']:>2}  "
              f"p50={r['p50_ms']:.2f}ms  thrpt={r['throughput_ips']:.1f} img/s  "
              f"macroF1={r['macro_f1']}  acc={r['accuracy']}", flush=True)
    print("[rung3b] appended to edge/results.csv + edge/RESULTS.md")


if __name__ == "__main__":
    main()
