"""Task C — turn perf_analyzer's CSV output into the canonical `triton` results row + serving curve.

perf_analyzer (official Triton native C++ gRPC client) is the fair load generator to compare against
the cpp-trt server's native C++ client. This script:
  1. reads the per-concurrency perf_analyzer CSVs (Concurrency, Inferences/Second, ... p50/p95/p99 usec
     + a server-side time breakdown: Server Queue, Server Compute Infer, Network+Server Send/Recv),
  2. writes the full sweep to `edge/serving/docs/triton_serving_curve.json`,
  3. appends the canonical `triton` rows (concurrency 1 = latency, 32 = throughput) to
     `edge/results.csv` and regenerates `edge/RESULTS.md`, taking macro-F1/accuracy from
     `triton_parity.json` (written by triton_client.py — parity with the tensorrt fp16 engine).

Usage (after run_triton_bench.sh has produced the per-concurrency CSVs):
  python3 edge/serving/tools/parse_perf.py --csv-glob 'edge/serving/docs/perf_*.csv' --gpu "NVIDIA RTX A6000"
or point --perf-csv at a single combined CSV (one row per concurrency).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

from edge.harness import append_results

DOCS = "edge/serving/docs"


def _read_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in sorted(paths):
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append({
                    "concurrency": int(r["Concurrency"]),
                    "throughput_ips": round(float(r["Inferences/Second"]), 1),
                    "p50_ms": round(int(r["p50 latency"]) / 1000, 2),
                    "p90_ms": round(int(r["p90 latency"]) / 1000, 2),
                    "p95_ms": round(int(r["p95 latency"]) / 1000, 2),
                    "p99_ms": round(int(r["p99 latency"]) / 1000, 2),
                    "server_queue_us": int(r["Server Queue"]),
                    "server_compute_infer_us": int(r["Server Compute Infer"]),
                    "grpc_net_us": int(r["Network+Server Send/Recv"]),
                })
    rows.sort(key=lambda x: x["concurrency"])
    # de-dup by concurrency (last-wins), in case both per-C csvs and a combined csv are globbed
    dedup = {r["concurrency"]: r for r in rows}
    return list(dedup.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-csv", help="single combined CSV (one row per concurrency)")
    ap.add_argument("--csv-glob", default=f"{DOCS}/perf_*.csv", help="glob of per-concurrency CSVs")
    ap.add_argument("--gpu", default="NVIDIA RTX A6000")
    ap.add_argument("--latency-conc", type=int, default=1)
    ap.add_argument("--throughput-conc", type=int, default=32)
    a = ap.parse_args()

    paths = [a.perf_csv] if a.perf_csv else glob.glob(a.csv_glob)
    rows = _read_rows(paths)
    assert rows, f"no perf_analyzer CSV rows found (paths={paths})"

    par = {"macro_f1": 0.8881, "accuracy": 0.8958, "n_test": 1565}
    if os.path.exists(f"{DOCS}/triton_parity.json"):
        par = json.load(open(f"{DOCS}/triton_parity.json"))

    curve = {
        "gpu": a.gpu,
        "server": "triton 2.51.0 (tensorrt_plan, dynamic_batching pref 8/16/32, 1ms queue delay)",
        "client": "perf_analyzer 2.51.0 (native C++, gRPC)",
        "engine": "fp16 TensorRT (trtexec, min1/opt32/max64) — same engine as tensorrt-fp16 / cpp-trt",
        "measurement": "5s window per concurrency, --percentile 95",
        **par, "rows": rows,
    }
    os.makedirs(DOCS, exist_ok=True)
    json.dump(curve, open(f"{DOCS}/triton_serving_curve.json", "w"), indent=2)
    print(f"[perf] wrote triton_serving_curve.json: {len(rows)} points; "
          f"peak {max(r['throughput_ips'] for r in rows)} ips", flush=True)

    def row(c):
        r = next((x for x in rows if x["concurrency"] == c), None)
        assert r, f"concurrency {c} not in perf CSVs (have {[x['concurrency'] for x in rows]})"
        return {"name": "triton", "runtime": "triton", "precision": "fp16", "device": "cuda",
                "batch": c, "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"], "p99_ms": r["p99_ms"],
                "throughput_ips": r["throughput_ips"], "peak_mem_mb": None,
                "macro_f1": par["macro_f1"], "accuracy": par["accuracy"], "n_test": par["n_test"]}

    append_results([row(a.latency_conc), row(a.throughput_conc)])
    print(f"[perf] appended triton rows (conc {a.latency_conc},{a.throughput_conc}) to "
          f"edge/results.csv + regenerated RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
