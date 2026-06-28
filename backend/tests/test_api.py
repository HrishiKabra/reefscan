"""Phase 5 async-pipeline contract tests (stub mode — no weights/Supabase/R2 needed).

Drives the real async path: POST /infer enqueues, GET /infer/{job_id} polls to completion,
and the response is validated against the frozen contract (frontend/lib/types.ts).
"""
import io
import os
import time

os.environ["REEFSCAN_STUB"] = "1"  # force synthetic inference before app import

from PIL import Image  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

SEGMENT_KEYS = {
    "segment_id", "mask_area_px", "bbox", "predicted_class", "prediction_set",
    "prediction_set_size", "confidence_scores", "coverage_pct",
}


def _png(w=600, h=380):
    b = io.BytesIO()
    Image.new("RGB", (w, h), (20, 80, 90)).save(b, format="PNG")
    return b.getvalue()


def _run_to_completion(client, job_id, tries=60):
    for _ in range(tries):
        resp = client.get(f"/infer/{job_id}").json()
        if resp["status"] == "complete":
            return resp
        if resp["status"] == "failed":
            raise AssertionError(f"job failed: {resp.get('error_message')}")
        time.sleep(0.1)
    raise AssertionError("job did not complete in time")


def test_health_stub_mode():
    with TestClient(app) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok"
        assert h["stub_mode"] is True  # no weights -> stub


def test_async_infer_contract():
    with TestClient(app) as client:
        r = client.post("/infer", files={"file": ("reef.png", _png(), "image/png")})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        assert client.get(f"/infer/{job_id}").json()["status"] in ("queued", "processing", "complete")

        data = _run_to_completion(client, job_id)

        assert data["image_width"] == 600 and data["image_height"] == 380
        assert data["model_version"]
        assert len(data["segments"]) >= 1
        for seg in data["segments"]:
            assert set(seg) == SEGMENT_KEYS
            assert set(seg["confidence_scores"]) == {"healthy", "bleached"}
            assert seg["predicted_class"] in ("healthy", "bleached")
            assert seg["prediction_set_size"] == len(seg["prediction_set"]) >= 1
            assert abs(sum(seg["confidence_scores"].values()) - 1.0) < 1e-3

        s = data["summary"]
        assert s["total_segments"] == len(data["segments"])
        assert abs(s["area_weighted"]["healthy_pct"] + s["area_weighted"]["bleached_pct"] - 100) < 0.2
        assert s["dominant_status"] in ("healthy", "bleached")
        # uncertain accounting must match the per-segment sets
        unc = sum(1 for x in data["segments"] if x["prediction_set_size"] > 1)
        assert s["uncertain_segments"] == unc


def test_unknown_job_404():
    with TestClient(app) as client:
        assert client.get("/infer/does-not-exist").status_code == 404


def test_infer_requires_input():
    with TestClient(app) as client:
        assert client.post("/infer").status_code == 400
