"""Phase 7 observability aggregation tests + read-endpoint fallbacks (no Supabase)."""
import os

os.environ["REEFSCAN_STUB"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from backend import observability  # noqa: E402
from backend.main import app  # noqa: E402

CLASSES = ("healthy", "bleached")


def _logs():
    # two days, varied set sizes / latencies / labels
    rows = []
    for i in range(10):
        rows.append({"ts": "2026-06-20T10:00:00Z", "prediction_set_size": 1 if i % 2 else 2,
                     "latency_ms": 100 + i * 10, "predicted_label": "healthy" if i < 7 else "bleached"})
    for i in range(10):
        rows.append({"ts": "2026-06-27T10:00:00Z", "prediction_set_size": 2 if i % 2 else 1,
                     "latency_ms": 200 + i * 10, "predicted_label": "bleached" if i < 6 else "healthy"})
    return rows


def test_rolling_set_size():
    out = observability.rolling_set_size(_logs())
    assert [r["date"] for r in out] == ["2026-06-20", "2026-06-27"]
    assert all(1.0 <= r["avg_set_size"] <= 2.0 for r in out)
    assert out[0]["n"] == 10


def test_latency_percentiles():
    out = observability.latency_percentiles(_logs())
    assert out[0]["p50"] <= out[0]["p95"]
    assert out[1]["p95"] >= out[1]["p50"] >= 200


def test_class_distribution_windows():
    cd = observability.class_distribution(_logs(), CLASSES)
    assert abs(cd["current"]["healthy"] + cd["current"]["bleached"] - 100) < 0.2
    assert cd["current_window"] and cd["baseline_window"]


def test_build_shape():
    o = observability.build(_logs(), CLASSES)
    assert set(o) == {"drift", "latency", "class_distribution", "total_logs"}
    assert o["total_logs"] == 20


def test_read_endpoints_empty_without_supabase():
    with TestClient(app) as client:
        assert client.get("/review-queue").json() == []
        assert client.get("/reef-locations").json() == []
        assert client.get("/reef-locations/x/snapshots").json() == []
        obs = client.get("/observability").json()
        assert obs["total_logs"] == 0 and obs["drift"] == []


def test_confirm_validates_label():
    with TestClient(app) as client:
        assert client.post("/review-queue/abc/confirm", json={"label": "nope"}).status_code == 400
        # valid label but no supabase -> ok False, still 200
        r = client.post("/review-queue/abc/confirm", json={"label": "healthy"})
        assert r.status_code == 200 and r.json()["ok"] is False
