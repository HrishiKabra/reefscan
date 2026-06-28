"""Pydantic models for the FROZEN inference response contract. Phase 5.

These mirror frontend/lib/types.ts exactly — the field names/shape are the integration
seam (Phase 6 mocks it, Phase 5 fills it). Do not rename fields without changing both.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

CoralClass = Literal["healthy", "bleached"]
JobStatus = Literal["queued", "processing", "complete", "failed"]


class Segment(BaseModel):
    segment_id: int
    mask_area_px: int
    bbox: list[int]  # [x0, y0, x1, y1] in source-image px
    predicted_class: CoralClass
    prediction_set: list[CoralClass]
    prediction_set_size: int
    confidence_scores: dict[str, float]  # {"healthy": .., "bleached": ..}
    coverage_pct: float


class AreaWeighted(BaseModel):
    healthy_pct: float
    bleached_pct: float


class Summary(BaseModel):
    total_segments: int
    area_weighted: AreaWeighted
    uncertain_segments: int
    dominant_status: CoralClass


class InferenceResponse(BaseModel):
    """Returned by GET /infer/{job_id}. When status != 'complete', segments/summary are
    empty/None and the client keeps polling."""
    job_id: str
    status: JobStatus
    processing_time_ms: int = 0
    image_url: str = ""
    model_version: str = ""
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    segments: list[Segment] = Field(default_factory=list)
    summary: Optional[Summary] = None
    error_message: Optional[str] = None


class SubmitResponse(BaseModel):
    """Returned by POST /infer — the async enqueue ack."""
    job_id: str
    status: JobStatus = "queued"


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    stub_mode: bool
    model_version: str
    last_inference_latency_ms: Optional[int] = None
