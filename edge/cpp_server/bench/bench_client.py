"""Phase 2 gate — benchmark the C++ server and slot a `cpp-trt` row into edge/results.csv.

Mirrors edge/harness.py's convention (warmup, sync-free wall timing, np.percentile, throughput).
Two things:
  1. F1 parity: push the full 1,565-image test set through POST /infer, argmax the logits, compute
     macro-F1/accuracy. Same engine as the `tensorrt fp16` row -> F1 MUST match it (a bug check).
  2. Concurrency sweep 1->8->16->32->64: p50/p95/p99 + throughput. Appends batch-1 and batched
     `cpp-trt` rows to edge/results.csv (regenerating RESULTS.md) + writes a serving-curve artifact.

Run on the box after the server is up (ENGINE_PATH=... ./build/reefscan_server &):
  PYTHONPATH=. python edge/cpp_server/bench/bench_client.py --url http://localhost:8000 --gpu "A40"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import httpx
import numpy as np

from edge.data import load_test
from edge.harness import _percentile, append_results

SWEEP = [1, 8, 16, 32, 64]


async def _infer(cl: httpx.AsyncClient, url: str, body: bytes) -> np.ndarray:
    r = await cl.post(url + "/infer", content=body, headers={"Content-Type": "application/octet-stream"})
    r.raise_for_status()
    return np.frombuffer(r.content, dtype=np.float32)  # [2]


async def _bounded(url: str, bodies: list[bytes], concurrency: int, timed: bool):
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    out: list[np.ndarray | None] = [None] * len(bodies)
    async with httpx.AsyncClient(timeout=60) as cl:
        async def one(i):
            async with sem:
                t = time.perf_counter()
                out[i] = await _infer(cl, url, bodies[i])
                if timed:
                    lat.append((time.perf_counter() - t) * 1000)
        await asyncio.gather(*(one(i) for i in range(len(bodies))))
    return out, lat


async def amain(a):
    test_x, y = load_test()
    bodies = [test_x[i].numpy().astype(np.float32).tobytes() for i in range(len(y))]
    print(f"[bench] {len(bodies)} test images -> {a.url}", flush=True)

    # health + warmup
    async with httpx.AsyncClient(timeout=30) as cl:
        assert (await cl.get(a.url + "/health")).status_code == 200, "server /health not ok"
        for i in range(10):
            await _infer(cl, a.url, bodies[i])

    # 1) F1 parity over the full test set
    out, _ = await _bounded(a.url, bodies, concurrency=32, timed=False)
    logits = np.stack(out)
    pred = logits.argmax(1)
    acc = float((pred == y).mean())
    from sklearn.metrics import f1_score
    f1 = float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))
    print(f"[bench] macro-F1={f1:.4f} acc={acc:.4f} (n={len(y)})", flush=True)

    # parity vs the tensorrt fp16 row already in results.csv
    import csv
    trt_f1 = None
    if os.path.exists("edge/results.csv"):
        for r in csv.DictReader(open("edge/results.csv")):
            if r["runtime"] == "tensorrt" and r["precision"] == "fp16":
                trt_f1 = float(r["macro_f1"]); break
    if trt_f1 is not None:
        ok = abs(f1 - trt_f1) < 5e-3
        print(f"[bench] F1 parity vs tensorrt fp16 ({trt_f1:.4f}): {'PASS' if ok else 'FAIL'}", flush=True)

    # 2) concurrency sweep
    N = a.requests
    idx = [i % len(bodies) for i in range(N)]
    sub = [bodies[i] for i in idx]
    rows = []
    print(f"    {'conc':>4} {'p50':>8} {'p95':>8} {'p99':>8} {'req/s':>9}", flush=True)
    for c in SWEEP:
        t0 = time.perf_counter()
        _, lat = await _bounded(a.url, sub, concurrency=c, timed=True)
        wall = time.perf_counter() - t0
        row = {"concurrency": c, "p50_ms": _percentile(lat, 50), "p95_ms": _percentile(lat, 95),
               "p99_ms": _percentile(lat, 99), "throughput_ips": round(N / wall, 1)}
        rows.append(row)
        print(f"    {c:>4} {row['p50_ms']:>8.2f} {row['p95_ms']:>8.2f} {row['p99_ms']:>8.2f} "
              f"{row['throughput_ips']:>9.1f}", flush=True)

    # serving-curve artifact
    os.makedirs("edge/cpp_server/docs", exist_ok=True)
    json.dump({"gpu": a.gpu, "server": "cpp-trt (hand-written queue)", "rows": rows},
              open("edge/cpp_server/docs/serving_curve.json", "w"), indent=2)

    # results.csv rows: concurrency 1 (latency) + 32 (throughput). NOTE: 'batch' column here = client
    # concurrency (the server batches dynamically) — documented in DECISIONS.md.
    def res_row(c):
        r = next(x for x in rows if x["concurrency"] == c)
        return {"name": "cpp-trt", "runtime": "cpp-trt", "precision": "fp16", "device": "cuda",
                "batch": c, "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"], "p99_ms": r["p99_ms"],
                "throughput_ips": r["throughput_ips"], "peak_mem_mb": None,
                "macro_f1": round(f1, 4), "accuracy": round(acc, 4), "n_test": len(y)}
    append_results([res_row(1), res_row(32)])
    print("[bench] appended cpp-trt rows to edge/results.csv + regenerated RESULTS.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--gpu", default="unknown")
    ap.add_argument("--requests", type=int, default=384, help="requests per concurrency level")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
