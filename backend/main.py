"""ReefScan FastAPI app — async-job inference. Phase 5.

  POST /infer            multipart file (image|video) or form `url` -> {job_id}  (enqueue)
  GET  /infer/{job_id}   -> InferenceResponse (status + results when complete)  (poll)
  GET  /health           -> model load status + last inference latency

Images and video share the SAME async path (CLAUDE.md). Models are loaded once at startup
and kept resident (Phase 1.5: RAM cleared, no lazy-load). The CPU-bound pipeline runs in a
threadpool so polling stays responsive. Phase 6 frontend swaps lib/api.ts onto these routes.
"""
from __future__ import annotations

import asyncio
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import observability
from .config import settings
from .inference import pipeline
from .jobs import JobStore, run_job
from .persistence import supabase
from .schemas import HealthResponse, InferenceResponse, SubmitResponse

logging.basicConfig(level=logging.INFO)
store = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load_models()  # resident SAM2 + DINOv2 (or stub fallback)
    yield


app = FastAPI(title="ReefScan", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = pipeline.models_loaded()
    return HealthResponse(
        status="ok",
        models_loaded=loaded,
        stub_mode=not loaded,
        model_version=pipeline.model_version(),
        last_inference_latency_ms=pipeline.last_latency_ms(),
    )


async def _read_input(file: Optional[UploadFile], url: Optional[str]) -> tuple[bytes, str]:
    """Return (bytes, kind) for an uploaded file or a url. kind in {'image','video'}."""
    if file is not None:
        data = await file.read()
        kind = "video" if (file.content_type or "").startswith("video") else "image"
        return data, kind
    if url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                r = await client.get(url)
                r.raise_for_status()
            ct = r.headers.get("content-type", "")
            return r.content, ("video" if ct.startswith("video") else "image")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"could not fetch url: {e}")
    raise HTTPException(status_code=400, detail="provide a file upload or a `url`")


@app.post("/infer", response_model=SubmitResponse, status_code=202)
async def infer(
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    reef_location_id: Optional[str] = Form(default=None),
) -> SubmitResponse:
    data, kind = await _read_input(file, url)
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    job_id = store.create(kind, reef_location_id, source_url=url or "")
    asyncio.create_task(run_job(store, job_id, data, kind, reef_location_id))
    return SubmitResponse(job_id=job_id)


@app.get("/infer/{job_id}", response_model=InferenceResponse)
async def get_job(job_id: str) -> InferenceResponse:
    resp = store.response(job_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return resp


# --- read endpoints (Phase 6 review/tracker + Phase 7 dashboard) ---
# All read from Supabase; return [] when Supabase creds are absent (the frontend keeps
# its own mock for that case, selected by NEXT_PUBLIC_REEFSCAN_API being unset).

@app.get("/review-queue")
async def review_queue(limit: int = 50) -> list[dict]:
    return [{
        "id": r.get("id"), "image_id": r.get("image_id"), "segment_id": r.get("segment_id"),
        "patch_url": r.get("patch_url") or "", "image_url": r.get("image_url") or "",
        "prediction_set": r.get("prediction_set") or [],
        "confidence_scores": {"healthy": r.get("conf_healthy") or 0.0,
                              "bleached": r.get("conf_bleached") or 0.0},
        "model_version": r.get("model_version") or "",
        "reef_location": r.get("reef_location_id") or "",
        "created_at": r.get("created_at") or "", "status": r.get("status") or "pending",
    } for r in supabase.review_queue(limit)]


@app.post("/review-queue/{review_id}/confirm")
async def confirm(review_id: str, label: str = Body(..., embed=True)) -> dict:
    if label not in settings.classes:
        raise HTTPException(status_code=400, detail=f"label must be one of {settings.classes}")
    ok = supabase.confirm_label(review_id, label)
    return {"ok": ok, "review_id": review_id, "label": label}


@app.get("/reef-locations")
async def reef_locations() -> list[dict]:
    return [{"id": r.get("id"), "name": r.get("name"),
             "lat": r.get("latitude"), "lng": r.get("longitude")}
            for r in supabase.reef_locations()]


@app.get("/reef-locations/{reef_id}/snapshots")
async def snapshots(reef_id: str) -> list[dict]:
    out = []
    for r in supabase.snapshots(reef_id):
        total_area = r.get("total_area_px") or 0
        healthy_pct = round((r.get("healthy_area_px") or 0) / total_area * 100, 1) if total_area else 0.0
        out.append({
            "date": str(r.get("snapshot_time", ""))[:10],
            "healthy_pct": healthy_pct, "bleached_pct": round(100 - healthy_pct, 1),
            "total_segments": r.get("total_segments") or 0,
            "avg_set_size": None,  # drift lives in /observability, not health_snapshots
        })
    return out


@app.get("/observability")
async def get_observability() -> dict:
    return observability.build(supabase.recent_logs(), settings.classes)
