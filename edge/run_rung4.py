"""Rung 4 — TensorRT (fp16 + int8 with entropy calibration). GPU-only; NOT locally verifiable.

Builds TensorRT engines from the Rung-3 fp32 ONNX and registers two variants:
  1. tensorrt-fp16 — FP16 engine. TRT fuses the whole transformer and autotunes kernels for the
     actual GPU; should match/beat ORT fp16. Lossless (correctness-gated vs PyTorch).
  2. tensorrt-int8 — INT8 engine calibrated with IInt8EntropyCalibrator2 over the REPRESENTATIVE
     train subsample (data.load_calibration() — invariant #3). This is the PAYOFF to the Rung-3b
     negative: ORT static int8 (MinMax/CPU) collapsed to 0.40 F1; TRT's entropy calibration + int8
     tensor-core kernels are the viable int8 path. We report whatever accuracy it actually recovers.

Targets the TensorRT 10.x Python API (execute_async_v3 / set_tensor_address). Run in the Colab
block with `tensorrt==10.5.0` pinned (see edge/colab/reefscan_edge.ipynb). Device buffers are torch
cuda tensors (no pycuda); input stays GPU-resident so timing is apples-to-apples with the other rungs.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from edge.data import load_calibration, load_test
from edge.harness import append_results, benchmark
from edge.model import load_model
from edge.run_rung3 import ONNX_DIR, ONNX_PATH, export_onnx

FP16_PLAN = os.path.join(ONNX_DIR, "dinov2_trt_fp16.plan")
INT8_PLAN = os.path.join(ONNX_DIR, "dinov2_trt_int8.plan")
CALIB_CACHE = os.path.join(ONNX_DIR, "dinov2_trt_int8_calib.cache")
CALIB_BATCH = 8
PROFILE = ((1, 3, 224, 224), (32, 3, 224, 224), (64, 3, 224, 224))  # min, opt, max (>=64 for eval)


def _calibrator(trt, calib: torch.Tensor, in_name: str):
    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.calib = calib
            self.idx = 0
            self.dev = torch.empty((CALIB_BATCH, 3, 224, 224), dtype=torch.float32, device="cuda")

        def get_batch_size(self):
            return CALIB_BATCH

        def get_batch(self, names):
            if self.idx + CALIB_BATCH > self.calib.shape[0]:
                return None
            self.dev.copy_(self.calib[self.idx:self.idx + CALIB_BATCH].cuda())
            self.idx += CALIB_BATCH
            return [int(self.dev.data_ptr())]

        def read_calibration_cache(self):
            return open(CALIB_CACHE, "rb").read() if os.path.exists(CALIB_CACHE) else None

        def write_calibration_cache(self, cache):
            with open(CALIB_CACHE, "wb") as f:
                f.write(cache)

    return EntropyCalibrator()


def build_engine(precision: str, calib: torch.Tensor | None = None) -> bytes:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 10: explicit-batch is default
    parser = trt.OnnxParser(network, logger)
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[rung4] ONNX parse error:", parser.get_error(i), flush=True)
            raise RuntimeError("TensorRT failed to parse the ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    in_name = network.get_input(0).name

    profile = builder.create_optimization_profile()
    profile.set_shape(in_name, *PROFILE)
    config.add_optimization_profile(profile)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # fp16 fallback for layers int8 can't take
        calib_profile = builder.create_optimization_profile()
        cb = (CALIB_BATCH, 3, 224, 224)
        calib_profile.set_shape(in_name, cb, cb, cb)
        config.set_calibration_profile(calib_profile)
        config.int8_calibrator = _calibrator(trt, calib, in_name)

    print(f"[rung4] building {precision} engine (this can take several minutes)...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT {precision} engine build failed")
    return bytes(serialized)


def load_or_build(precision: str, plan_path: str, calib: torch.Tensor | None = None):
    import tensorrt as trt
    if os.path.exists(plan_path):
        print(f"[rung4] loading cached engine {plan_path}", flush=True)
        data = open(plan_path, "rb").read()
    else:
        data = build_engine(precision, calib)
        with open(plan_path, "wb") as f:
            f.write(data)
        print(f"[rung4] saved {plan_path} ({len(data) / 1e6:.0f} MB)", flush=True)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return runtime.deserialize_cuda_engine(data)


def make_predict(engine):
    import tensorrt as trt
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

    def predict(x):  # x: torch cuda fp32, already resident (no H2D in the timed call)
        x = x.contiguous()
        bs = x.shape[0]
        context.set_input_shape(in_name, (bs, 3, 224, 224))
        out = torch.empty((bs, 2), dtype=torch.float32, device="cuda")
        context.set_tensor_address(in_name, int(x.data_ptr()))
        context.set_tensor_address(out_name, int(out.data_ptr()))
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return out  # torch cuda tensor; harness syncs device-wide before reading

    return predict


def main():
    if not torch.cuda.is_available():
        raise SystemExit("[rung4] TensorRT requires a CUDA GPU — run this on Colab (L4).")
    import tensorrt as trt
    print(f"[rung4] TensorRT {trt.__version__}  |  {torch.cuda.get_device_name(0)}", flush=True)

    model = load_model(device="cuda")
    test_x, test_y = load_test()
    if not os.path.exists(ONNX_PATH):
        export_onnx(model, "cuda")

    rows = []

    # fp16
    eng16 = load_or_build("fp16", FP16_PLAN)
    pred16 = make_predict(eng16)
    with torch.no_grad():
        xb = test_x[:8].cuda()
        ref = model(xb).float().cpu().numpy()
    agree = int((pred16(xb).cpu().numpy().argmax(1) == ref.argmax(1)).sum())
    print(f"[rung4] fp16 correctness gate: argmax agreement {agree}/8 vs PyTorch", flush=True)
    assert agree >= 7, "TRT fp16 disagrees with PyTorch — engine is wrong"
    rows += benchmark("tensorrt-fp16", "tensorrt", "fp16", pred16, test_x, test_y, "cuda")

    # int8 (entropy-calibrated)
    eng8 = load_or_build("int8", INT8_PLAN, calib=load_calibration())
    rows += benchmark("tensorrt-int8", "tensorrt", "int8", make_predict(eng8), test_x, test_y, "cuda")

    append_results(rows)
    for r in rows:
        print(f"[rung4] {r['name']:<14} batch={r['batch']:>2}  p50={r['p50_ms']:.2f}ms  "
              f"p95={r['p95_ms']:.2f}ms  thrpt={r['throughput_ips']:.1f} img/s  "
              f"macroF1={r['macro_f1']}  acc={r['accuracy']}", flush=True)
    f1_int8 = next(r["macro_f1"] for r in rows if r["name"] == "tensorrt-int8")
    f1_fp16 = next(r["macro_f1"] for r in rows if r["name"] == "tensorrt-fp16")
    print(f"[rung4] int8 vs fp16 accuracy: {f1_int8} vs {f1_fp16}  "
          f"(ORT static int8 was 0.3992 — TRT entropy {'recovers' if f1_int8 > 0.8 else 'still drops'})",
          flush=True)
    print("[rung4] appended to edge/results.csv + edge/RESULTS.md")


if __name__ == "__main__":
    main()
