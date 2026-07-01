"""Load test the ReefScan /infer serving path — p50/p95/p99 + throughput under concurrency.

Fires N requests at a target concurrency against the REAL async-job endpoint (POST /infer ->
poll GET /infer/{job_id} until `complete`), measuring END-TO-END request latency. Sweeps
concurrency so you can watch p99 diverge from p50 as the single worker + threadpool saturate —
the classic serving-under-load signal a running-mean latency would hide.

The image is a synthetic in-memory PNG, so there are no external deps or fixtures.

Run the backend first (stub is fine and fast — no GPU):
    REEFSCAN_STUB=1 uvicorn backend.main:app --port 8000
    python -m backend.loadtest --sweep 1,4,8,16,32 --n 64 --out docs/eval/loadtest.json
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from backend.observability import _percentile

_IMG_BYTES: bytes = b""


def _synthetic_png() -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


async def _one_request(client: httpx.AsyncClient, base: str, poll_s: float, timeout_s: float) -> float | None:
    """Submit -> poll to completion. Returns end-to-end latency in ms, or None on failure/timeout."""
    t0 = time.perf_counter()
    r = await client.post(f"{base}/infer", files={"file": ("s.png", _IMG_BYTES, "image/png")})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    deadline = t0 + timeout_s
    while time.perf_counter() < deadline:
        jr = await client.get(f"{base}/infer/{job_id}")
        st = jr.json().get("status")
        if st == "complete":
            return (time.perf_counter() - t0) * 1000.0
        if st == "failed":
            return None
        await asyncio.sleep(poll_s)
    return None


async def _run_level(base: str, n: int, concurrency: int, poll_s: float, timeout_s: float) -> dict:
    sem = asyncio.Semaphore(concurrency)
    lats: list[float] = []

    async with httpx.AsyncClient(timeout=timeout_s + 5) as client:
        async def worker():
            async with sem:
                lat = await _one_request(client, base, poll_s, timeout_s)
                if lat is not None:
                    lats.append(lat)

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(n)])
        wall = time.perf_counter() - t0

    return {
        "concurrency": concurrency, "n": n, "ok": len(lats),
        "p50_ms": _percentile(lats, 50), "p95_ms": _percentile(lats, 95), "p99_ms": _percentile(lats, 99),
        "throughput_rps": round(len(lats) / wall, 2) if wall > 0 else 0.0, "wall_s": round(wall, 2),
    }


async def _amain(a) -> None:
    global _IMG_BYTES
    _IMG_BYTES = _synthetic_png()
    levels = [int(x) for x in a.sweep.split(",")]

    # warm the server (model load / first-hit costs) so level 1 isn't skewed
    async with httpx.AsyncClient(timeout=a.timeout + 5) as client:
        await _one_request(client, a.base_url, a.poll, a.timeout)

    print(f"[loadtest] {a.base_url}  n={a.n} per level  sweep={levels}  poll={a.poll}s", flush=True)
    print(f"    {'conc':>4} {'ok':>5} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9} {'req/s':>8} {'wall s':>7}", flush=True)
    rows = []
    for c in levels:
        row = await _run_level(a.base_url, a.n, c, a.poll, a.timeout)
        rows.append(row)
        print(f"    {row['concurrency']:>4} {row['ok']:>5} {row['p50_ms']:>9.1f} {row['p95_ms']:>9.1f} "
              f"{row['p99_ms']:>9.1f} {row['throughput_rps']:>8.1f} {row['wall_s']:>7.1f}", flush=True)

    p99s = [r["p99_ms"] for r in rows]
    ratios = [r["p99_ms"] / max(r["p50_ms"], 1e-9) for r in rows]
    imax = ratios.index(max(ratios))
    peak = max(rows, key=lambda r: r["throughput_rps"])
    print(f"[loadtest] p99 {p99s[0]:.0f}ms (conc {levels[0]}) -> {max(p99s):.0f}ms peak; "
          f"widest p99/p50 gap {max(ratios):.2f}x @ conc {levels[imax]}; "
          f"throughput peaks {peak['throughput_rps']:.0f} req/s @ conc {peak['concurrency']} "
          f"then saturates", flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"target": a.base_url, "machine": a.machine, "stub": a.stub, "poll_s": a.poll, "levels": rows},
            indent=2))
        print(f"[loadtest] wrote {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--sweep", default="1,4,8,16,32", help="comma-separated concurrency levels")
    ap.add_argument("--n", type=int, default=64, help="requests per concurrency level")
    ap.add_argument("--poll", type=float, default=0.05, help="poll interval (s) while awaiting a job")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-request timeout (s)")
    ap.add_argument("--out", default=None, help="write a JSON summary (e.g. docs/eval/loadtest.json)")
    ap.add_argument("--machine", default="local", help="label recorded in the JSON (e.g. 'M-series CPU')")
    ap.add_argument("--stub", action="store_true", help="record that the backend ran in stub mode")
    asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    main()
