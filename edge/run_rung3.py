"""Rung 3 — ONNX export + ONNX Runtime (CUDA EP), fp32. Registers the onnxruntime variant.

Story: drop PyTorch at serving time. Export the trained model to ONNX (opset 17, dynamic batch
axis; H/W stay static at 224 — every pipeline input is a 224 crop), verify ORT logits match
PyTorch (correctness gate), then benchmark onnxruntime-gpu. This is also the on-ramp to
TensorRT/Triton (Rungs 4-5).

Fairness (matches the PyTorch baseline timing exactly):
  - GPU: IO-binding keeps the input tensor resident on the GPU (bound from the torch cuda
    tensor's data_ptr), so NO host->device copy is inside the timed call — same as PyTorch,
    where x is `.to(device)` once outside the loop. Only the tiny (B x 2) output is copied out.
  - The harness syncs device-wide (torch.cuda.synchronize covers ORT's stream), so timing is real.
  - We HARD-FAIL if the CUDA EP didn't actually load (ORT silently falls back to CPU otherwise),
    so a 'cuda' row can never be mislabeled CPU latency.

Local (CPU): python -m edge.run_rung3   -> exports + correctness gate via onnxruntime CPU.
Colab (GPU): same command on a CUDA box -> the real Rung-3 numbers (needs onnxruntime-gpu).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from edge.data import load_test
from edge.harness import append_results, benchmark
from edge.model import load_model

ONNX_DIR = os.environ.get("REEFSCAN_ONNX_DIR", "edge/artifacts")
ONNX_PATH = os.path.join(ONNX_DIR, "dinov2_fp32.onnx")


def export_onnx(model, device: str) -> None:
    os.makedirs(ONNX_DIR, exist_ok=True)
    dummy = torch.randn(2, 3, 224, 224, device=device)  # batch 2 so the dynamic axis is exercised
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, ONNX_PATH,
            input_names=["pixel_values"], output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, do_constant_folding=True,
        )
    print(f"[rung3] exported ONNX -> {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1e6:.0f} MB)", flush=True)


def make_session(device: str, path: str = ONNX_PATH):
    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(path, providers=providers)
    active = sess.get_providers()
    print(f"[rung3] ORT active providers: {active}", flush=True)
    if device == "cuda" and "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            "CUDAExecutionProvider not active — onnxruntime-gpu / CUDA libs failed to load. "
            "Aborting so CPU latency is never mislabeled as 'cuda'. "
            "Check the LD_LIBRARY_PATH setup cell (point ORT at torch's bundled CUDA/cuDNN).")
    return sess


def make_predict(sess, device: str):
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    if device == "cuda":
        def predict(x):  # x: torch cuda tensor, already resident (no H2D in the timed call)
            x = x.contiguous()
            io = sess.io_binding()
            io.bind_input(in_name, "cuda", 0, np.float32, tuple(x.shape), x.data_ptr())
            io.bind_output(out_name, "cuda", 0)
            sess.run_with_iobinding(io)
            return io.copy_outputs_to_cpu()[0]  # tiny (B x 2)
        return predict

    def predict(x):
        return sess.run([out_name], {in_name: x.detach().cpu().numpy().astype(np.float32)})[0]
    return predict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rung3] device={device}", flush=True)
    model = load_model(device=device)
    test_x, test_y = load_test()
    print(f"[rung3] test set: {tuple(test_x.shape)}  labels={len(test_y)}", flush=True)

    if os.path.exists(ONNX_PATH):
        print(f"[rung3] reusing {ONNX_PATH}", flush=True)
    else:
        export_onnx(model, device)

    sess = make_session(device)
    predict = make_predict(sess, device)

    # Correctness gate (invariant in spirit): ORT must match PyTorch on a real batch.
    with torch.no_grad():
        xb = test_x[:8].to(device)
        torch_logits = model(xb).detach().float().cpu().numpy()
    max_diff = float(np.abs(torch_logits - predict(xb)).max())
    print(f"[rung3] correctness gate: max|ORT - PyTorch| = {max_diff:.2e}", flush=True)
    assert max_diff < 1e-2, f"ONNX logits diverge from PyTorch ({max_diff}) — export is wrong"
    print("[rung3] correctness gate PASSED", flush=True)

    rows = benchmark("onnxruntime", "onnxruntime", "fp32", predict, test_x, test_y, device)
    append_results(rows)
    for r in rows:
        print(f"[rung3] batch={r['batch']:>2}  p50={r['p50_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  "
              f"thrpt={r['throughput_ips']:.1f} img/s  macroF1={r['macro_f1']}  acc={r['accuracy']}", flush=True)
    print("[rung3] appended to edge/results.csv + edge/RESULTS.md")


if __name__ == "__main__":
    main()
