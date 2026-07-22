"""Task C — drive real NVIDIA Triton over gRPC to (1) prove F1 parity with the TensorRT fp16
engine and (2) record a `triton` row directly comparable to the `cpp-trt` hand-written server.

Same discipline as edge/harness.py and edge/cpp_server/bench/bench_client.py:
  - The SAME 1,565-image test set. macro-F1 MUST match the `tensorrt fp16` row (parity, not a
    new number) — a mismatch means the engine/repo is wrong, not a finding.
  - Batch-1 requests on the wire; Triton's `dynamic_batching` coalesces them server-side — the
    exact same setup as the cpp server, so the `batch` column here is CLIENT CONCURRENCY (not the
    inference batch size), matching the cpp-trt rows. Latency is end-to-end gRPC (network + Triton
    dynamic batcher + TensorRT). This keeps triton and cpp-trt on the same axis.
  - Warmup, then per-request wall-clock timing; np.percentile; wall throughput.

perf_analyzer (run alongside in run_triton.sh) is the industry-standard load generator; this client
adds the F1 parity check perf_analyzer can't do (it sends random data) and a row on our own schema.

Run on the box after tritonserver is READY:
  PYTHONPATH=. python edge/serving/tools/triton_client.py --gpu "NVIDIA RTX A6000"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from edge.data import load_test
from edge.harness import append_results

MODEL = "reefscan_dinov2"
SWEEP = [1, 8, 16, 32, 64]
_local = threading.local()


def _client(url: str):
    import tritonclient.grpc as grpcclient
    if not hasattr(_local, "cl"):
        _local.cl = grpcclient.InferenceServerClient(url=url, verbose=False)
    return _local.cl


def _infer(url: str, img: np.ndarray) -> np.ndarray:
    import tritonclient.grpc as grpcclient
    cl = _client(url)
    inp = grpcclient.InferInput("pixel_values", [1, 3, 224, 224], "FP32")
    inp.set_data_from_numpy(img[None].astype(np.float32))
    out = grpcclient.InferRequestedOutput("logits")
    r = cl.infer(model_name=MODEL, inputs=[inp], outputs=[out])
    return r.as_numpy("logits")[0]  # [2]


def _pct(v, p):
    return round(float(np.percentile(v, p)), 2) if len(v) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--gpu", default="unknown")
    ap.add_argument("--requests", type=int, default=768, help="requests per concurrency level")
    a = ap.parse_args()

    test_x, y = load_test()
    imgs = [test_x[i].numpy().astype(np.float32) for i in range(len(y))]
    print(f"[triton] {len(imgs)} test images -> {a.url}", flush=True)

    # warmup
    for i in range(10):
        _infer(a.url, imgs[i])

    # 1) F1 parity over the full test set (32-way concurrent, untimed)
    with ThreadPoolExecutor(max_workers=32) as ex:
        logits = list(ex.map(lambda im: _infer(a.url, im), imgs))
    pred = np.stack(logits).argmax(1)
    acc = float((pred == y).mean())
    from sklearn.metrics import f1_score
    f1 = float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))
    print(f"[triton] macro-F1={f1:.4f} acc={acc:.4f} (n={len(y)})", flush=True)

    trt_f1 = None
    if os.path.exists("edge/results.csv"):
        for r in csv.DictReader(open("edge/results.csv")):
            if r["runtime"] == "tensorrt" and r["precision"] == "fp16":
                trt_f1 = float(r["macro_f1"]); break
    if trt_f1 is not None:
        ok = abs(f1 - trt_f1) < 5e-3
        print(f"[triton] F1 parity vs tensorrt fp16 ({trt_f1:.4f}): {'PASS' if ok else 'FAIL'}", flush=True)

    # 2) concurrency sweep — batch-1 requests, Triton batches dynamically
    N = a.requests
    sub = [imgs[i % len(imgs)] for i in range(N)]
    rows = []
    print(f"    {'conc':>4} {'p50':>8} {'p95':>8} {'p99':>8} {'req/s':>9}", flush=True)
    for c in SWEEP:
        lat: list[float] = []
        lock = threading.Lock()

        def one(im):
            t = time.perf_counter()
            _infer(a.url, im)
            dt = (time.perf_counter() - t) * 1000
            with lock:
                lat.append(dt)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            list(ex.map(one, sub))
        wall = time.perf_counter() - t0
        row = {"concurrency": c, "p50_ms": _pct(lat, 50), "p95_ms": _pct(lat, 95),
               "p99_ms": _pct(lat, 99), "throughput_ips": round(N / wall, 1)}
        rows.append(row)
        print(f"    {c:>4} {row['p50_ms']:>8.2f} {row['p95_ms']:>8.2f} {row['p99_ms']:>8.2f} "
              f"{row['throughput_ips']:>9.1f}", flush=True)

    os.makedirs("edge/serving/docs", exist_ok=True)
    json.dump({"gpu": a.gpu, "server": "triton (dynamic_batching)", "rows": rows},
              open("edge/serving/docs/triton_serving_curve.json", "w"), indent=2)

    def res_row(c):
        r = next(x for x in rows if x["concurrency"] == c)
        return {"name": "triton", "runtime": "triton", "precision": "fp16", "device": "cuda",
                "batch": c, "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"], "p99_ms": r["p99_ms"],
                "throughput_ips": r["throughput_ips"], "peak_mem_mb": None,
                "macro_f1": round(f1, 4), "accuracy": round(acc, 4), "n_test": len(y)}
    append_results([res_row(1), res_row(32)])
    print("[triton] appended triton rows to edge/results.csv + regenerated RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
