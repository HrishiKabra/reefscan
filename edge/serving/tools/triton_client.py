"""Task C — drive real NVIDIA Triton over gRPC to prove **F1 parity** with the TensorRT fp16
engine (the one thing `perf_analyzer` can't do — it sends random data).

Division of labour for the `triton` benchmark:
  - **This client** owns correctness: push the SAME 1,565-image test set through Triton, argmax
    the logits, and assert macro-F1 matches the `tensorrt fp16` row (parity, not a new number —
    a mismatch means the engine/repo is wrong). It writes `triton_parity.json` for the row.
  - **`perf_analyzer`** (official native C++ client, run in `run_triton_bench.sh` + parsed by
    `parse_perf.py`) owns the canonical latency/throughput row — a fair native-client number vs
    the cpp-trt server's native client. A Python client can't match a C++ client's throughput.
  - This client ALSO runs a Python-threaded sweep, saved as a **cross-check** curve
    (`triton_pyclient_curve.json`). Its latency tracks perf_analyzer at low concurrency but its
    throughput is client-bound under load (GIL) — which is exactly why perf_analyzer is canonical.
    It is NOT written to results.csv.

Batch-1 requests on the wire; Triton's `dynamic_batching` coalesces them server-side — same setup
as the cpp server, so "concurrency" here is the comparable axis (not inference batch size).

Run on the box after tritonserver is READY:
  PYTHONPATH=. python3 edge/serving/tools/triton_client.py --gpu "NVIDIA RTX A6000"
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

    # parity.json — consumed by parse_perf.py to fill the macro-F1/acc columns of the canonical row
    os.makedirs("edge/serving/docs", exist_ok=True)
    json.dump({"macro_f1": round(f1, 4), "accuracy": round(acc, 4), "n_test": len(y)},
              open("edge/serving/docs/triton_parity.json", "w"), indent=2)

    # 2) Python cross-check sweep (NOT canonical — perf_analyzer owns the results.csv row).
    #    Batch-1 requests, Triton batches dynamically. Throughput here is client-bound under load.
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

    json.dump({"gpu": a.gpu, "client": "python tritonclient (threaded) — CROSS-CHECK ONLY",
               "note": "latency tracks perf_analyzer at low concurrency; throughput is GIL-bound "
                       "under load. Canonical row is perf_analyzer (see triton_serving_curve.json).",
               "macro_f1": round(f1, 4), "accuracy": round(acc, 4), "rows": rows},
              open("edge/serving/docs/triton_pyclient_curve.json", "w"), indent=2)
    print("[triton] wrote triton_parity.json + triton_pyclient_curve.json (cross-check). "
          "Canonical latency/throughput row comes from perf_analyzer via parse_perf.py.", flush=True)


if __name__ == "__main__":
    main()
