"""Observability aggregations over inference_logs. Phase 7 (+ serving metrics).

Pure functions (testable) that turn raw inference_logs rows into the dashboard views
computed entirely from Supabase data — no external observability tool:
  - rolling mean prediction_set_size per day  -> drift proxy (rising = distribution shift)
  - latency p50 / p95 / p99 per day
  - latency_summary: p50/p95/p99 + throughput (req/s) over the whole recent window
  - class distribution: current 7-day window vs the prior 7-day baseline
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta


def _day(ts) -> str:
    return str(ts)[:10]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * p / 100.0
    f = int(k)
    if f + 1 < len(v):
        return round(v[f] + (v[f + 1] - v[f]) * (k - f), 1)
    return round(v[f], 1)


def rolling_set_size(logs: list[dict]) -> list[dict]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in logs:
        by[_day(r["ts"])].append(float(r.get("prediction_set_size", 1)))
    return [{"date": d, "avg_set_size": round(sum(v) / len(v), 3), "n": len(v)}
            for d, v in sorted(by.items())]


def latency_percentiles(logs: list[dict]) -> list[dict]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in logs:
        by[_day(r["ts"])].append(float(r.get("latency_ms", 0)))
    return [{"date": d, "p50": _percentile(v, 50), "p95": _percentile(v, 95),
             "p99": _percentile(v, 99), "n": len(v)}
            for d, v in sorted(by.items())]


def _parse_ts(ts) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def latency_summary(logs: list[dict]) -> dict:
    """p50/p95/p99 + throughput (req/s) over the whole recent window. Percentiles are over the
    window (not a running mean); throughput = requests / observed time-span."""
    lat = [float(r.get("latency_ms", 0)) for r in logs]
    tss = [t for t in (_parse_ts(r.get("ts")) for r in logs) if t is not None]
    span_s = (max(tss) - min(tss)).total_seconds() if len(tss) >= 2 else 0.0
    return {
        "p50_ms": _percentile(lat, 50), "p95_ms": _percentile(lat, 95), "p99_ms": _percentile(lat, 99),
        "throughput_rps": round(len(logs) / span_s, 3) if span_s > 0 else 0.0,
        "n": len(logs), "window_s": round(span_s, 1),
    }


def class_distribution(logs: list[dict], classes: tuple[str, ...]) -> dict:
    days = sorted({_day(r["ts"]) for r in logs})
    if not days:
        return {"current": {}, "baseline": {}, "current_window": None, "baseline_window": None}
    anchor = date.fromisoformat(days[-1])
    cur_lo = anchor - timedelta(days=6)
    base_hi = cur_lo - timedelta(days=1)
    base_lo = base_hi - timedelta(days=6)

    def frac(lo: date, hi: date) -> dict:
        counts = {c: 0 for c in classes}
        for r in logs:
            d = date.fromisoformat(_day(r["ts"]))
            if lo <= d <= hi:
                lab = r.get("predicted_label")
                if lab in counts:
                    counts[lab] += 1
        total = sum(counts.values()) or 1
        return {c: round(counts[c] / total * 100, 1) for c in classes}

    return {
        "current": frac(cur_lo, anchor),
        "baseline": frac(base_lo, base_hi),
        "current_window": [cur_lo.isoformat(), anchor.isoformat()],
        "baseline_window": [base_lo.isoformat(), base_hi.isoformat()],
    }


def build(logs: list[dict], classes: tuple[str, ...]) -> dict:
    return {
        "drift": rolling_set_size(logs),
        "latency": latency_percentiles(logs),
        "latency_summary": latency_summary(logs),
        "class_distribution": class_distribution(logs, classes),
        "total_logs": len(logs),
    }
