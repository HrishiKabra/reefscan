"""Profiling rung — where does the 2.24 ms go? Layer-level profile of the winning TRT fp16 engine.

Attaches a TensorRT IProfiler to the execution context, runs the fp16 engine, and aggregates
per-layer GPU time. Answers two questions a systems-perf interviewer will ask:
  1. WHERE is the time? (top kernels by total ms — the things worth optimizing next)
  2. Did TRT actually fuse the transformer? (layer count: the fp32 ONNX has ~hundreds of ops;
     TRT collapses them into a handful of fused ForeignNode kernels.)

Outputs edge/docs/profile_trt.png (top-K bar chart) + edge/docs/profile_trt.json (full breakdown).
GPU-only; reuses the Rung-4 fp16 engine (.plan). Run after run_rung4.

Usage (Colab, after Rung 4): PYTHONPATH=. python -m edge.profile_trt
"""
from __future__ import annotations

import json
import os

import torch

from edge.run_rung4 import FP16_PLAN, load_or_build

OUT_DIR = "edge/docs"
PNG = os.path.join(OUT_DIR, "profile_trt.png")
JSON = os.path.join(OUT_DIR, "profile_trt.json")
BATCH = 32
WARMUP = 10
ITERS = 50


def main():
    if not torch.cuda.is_available():
        raise SystemExit("[profile] needs a CUDA GPU — run on Colab after Rung 4.")
    import tensorrt as trt
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(FP16_PLAN):
        raise SystemExit(f"[profile] {FP16_PLAN} missing — run `python -m edge.run_rung4` first.")
    engine = load_or_build("fp16", FP16_PLAN)
    context = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

    x = torch.randn(BATCH, 3, 224, 224, device="cuda")
    out = torch.empty(BATCH, 2, dtype=torch.float32, device="cuda")
    context.set_input_shape(in_name, (BATCH, 3, 224, 224))
    # IProfiler is only reliable with the SYNCHRONOUS execute path (execute_v2 with a bindings
    # list ordered by IO-tensor index); the async path can report nothing.
    addr = {in_name: int(x.data_ptr()), out_name: int(out.data_ptr())}
    bindings = [addr[engine.get_tensor_name(i)] for i in range(engine.num_io_tensors)]

    class LayerProfiler(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.times: dict[str, float] = {}

        def report_layer_time(self, layer_name, ms):
            self.times[layer_name] = self.times.get(layer_name, 0.0) + ms

    for _ in range(WARMUP):  # warmup WITHOUT profiler (autotune/alloc shouldn't pollute)
        context.execute_v2(bindings)
    torch.cuda.synchronize()

    prof = LayerProfiler()
    context.profiler = prof
    for _ in range(ITERS):
        context.execute_v2(bindings)
    torch.cuda.synchronize()

    per_iter = {k: v / ITERS for k, v in prof.times.items()}
    total = sum(per_iter.values())
    ranked = sorted(per_iter.items(), key=lambda kv: kv[1], reverse=True)
    if not per_iter or total <= 0:
        raise SystemExit("[profile] IProfiler returned no per-layer data on this build "
                         "(engine may be a single fused block) — core results are unaffected.")

    print(f"[profile] TRT fp16 engine @ batch={BATCH}: {len(per_iter)} fused layers, "
          f"summed kernel time {total:.2f} ms/iter ({total / BATCH:.3f} ms/img)", flush=True)
    print(f"[profile] top kernels (ms/iter, % of total):", flush=True)
    for name, ms in ranked[:12]:
        short = (name[:70] + "…") if len(name) > 71 else name
        print(f"    {ms:7.3f}  {100 * ms / total:5.1f}%  {short}", flush=True)

    with open(JSON, "w") as f:
        json.dump({"batch": BATCH, "n_layers": len(per_iter), "total_ms_per_iter": total,
                   "ms_per_img": total / BATCH, "layers": ranked}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = ranked[:12][::-1]
        labels = [(n[:48] + "…") if len(n) > 49 else n for n, _ in top]
        vals = [v for _, v in top]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(labels, vals, color="#2166ac", alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f} ms ({100 * v / total:.0f}%)", va="center", fontsize=8)
        ax.set_xlabel("GPU time (ms/iter, batch=%d)  —  summed total %.2f ms" % (BATCH, total))
        ax.set_title(f"TensorRT fp16 engine — top kernels ({len(per_iter)} fused layers, L4)")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PNG, dpi=140, bbox_inches="tight")
        print(f"[profile] wrote {PNG} + {JSON}", flush=True)
    except Exception as e:  # noqa: BLE001 — plotting is optional, the json is the artifact
        print(f"[profile] (plot skipped: {e}); wrote {JSON}", flush=True)


if __name__ == "__main__":
    main()
