"""Full inference pipeline orchestration. Phase 5.

enhance -> frames (video) / passthrough (image) -> SAM2 AMG segment -> per-segment bbox
crop -> DINOv2 + conformal classify -> assemble the frozen contract -> log to Supabase +
store source to R2 -> return result dict.

Returns the InferenceResponse fields (minus job_id/status, which the JobStore adds).
Runs synchronously in a threadpool (called via run_in_executor) — CPU-bound.
"""
from __future__ import annotations

import logging
import time

from ..config import settings
from ..persistence import new_request_id, r2, supabase
from . import classifier, enhancer, frame_extractor, segmenter

logger = logging.getLogger(__name__)

_last_latency_ms: int | None = None


def load_models() -> None:
    """Load SAM2 + DINOv2 once at startup (resident, no lazy-load — Phase 1.5). On any
    failure (e.g. weights not yet on the Hub) we log and fall back to STUB mode so the
    service still answers with contract-valid output."""
    if settings.stub_mode:
        logger.warning("REEFSCAN_STUB=1 -> running inference in STUB mode")
        return
    try:
        segmenter.load()
        classifier.load()
    except Exception as e:  # noqa: BLE001
        logger.warning("model load failed (%s) -> STUB mode (contract-valid synthetic output)", e)


def models_loaded() -> bool:
    return segmenter.is_loaded() and classifier.is_loaded()


def model_version() -> str:
    return classifier.model_version()


def last_latency_ms() -> int | None:
    return _last_latency_ms


def run(job_id: str, data: bytes, kind: str, reef_location_id: str | None) -> dict:
    global _last_latency_ms
    t0 = time.perf_counter()
    request_id = new_request_id()

    content_type = "video/mp4" if kind == "video" else "image/jpeg"
    image_url = r2.upload(f"{request_id}/source", data, content_type)

    frames = frame_extractor.extract(data, kind)
    img = enhancer.enhance(frames[0])  # scaffold: classify on first frame (N-frame agg = future)
    W, H = img.size

    raw = segmenter.segment(img)
    q = classifier.qhat()
    total_area = sum(s["mask_area_px"] for s in raw) or 1

    segments: list[dict] = []
    for i, s in enumerate(raw, start=1):
        crop = img.crop(tuple(s["bbox"]))
        probs = classifier.classify(crop)
        pset, size = classifier.conformal_set(probs, q)
        predicted = max(probs, key=probs.get)
        segments.append({
            "segment_id": i,
            "mask_area_px": s["mask_area_px"],
            "bbox": s["bbox"],
            "predicted_class": predicted,
            "prediction_set": pset,
            "prediction_set_size": size,
            "confidence_scores": {k: round(v, 4) for k, v in probs.items()},
            "coverage_pct": round(s["mask_area_px"] / total_area * 100, 1),
        })

    summary = _summarize(segments, total_area)
    latency = int((time.perf_counter() - t0) * 1000)
    _last_latency_ms = latency
    version = classifier.model_version()

    _log_all(request_id, image_url, reef_location_id, segments, summary,
             total_area, latency, version)

    return {
        "processing_time_ms": latency,
        "image_url": image_url,
        "model_version": version,
        "image_width": W,
        "image_height": H,
        "segments": segments,
        "summary": summary,
    }


def _summarize(segments: list[dict], total_area: int) -> dict:
    healthy_area = sum(s["mask_area_px"] for s in segments if s["predicted_class"] == "healthy")
    healthy_pct = round(healthy_area / total_area * 100, 1) if total_area else 0.0
    return {
        "total_segments": len(segments),
        "area_weighted": {"healthy_pct": healthy_pct, "bleached_pct": round(100 - healthy_pct, 1)},
        "uncertain_segments": sum(1 for s in segments if s["prediction_set_size"] > 1),
        "dominant_status": "healthy" if healthy_pct >= 50 else "bleached",
    }


def _log_all(request_id, image_url, reef_id, segments, summary, total_area, latency, version):
    """Best-effort Supabase logging (no-op without creds). Mirrors backend/db/schema.sql."""
    inf_rows, review_rows = [], []
    counts = {"healthy": 0, "bleached": 0}
    areas = {"healthy": 0, "bleached": 0}
    for s in segments:
        cs = s["confidence_scores"]
        inf_rows.append({
            "request_id": request_id, "image_id": image_url, "segment_id": s["segment_id"],
            "reef_location_id": reef_id, "latency_ms": latency,
            "conf_healthy": cs.get("healthy", 0.0), "conf_bleached": cs.get("bleached", 0.0),
            "conf_dead": 0.0, "conf_algae_covered": 0.0,
            "prediction_set": s["prediction_set"], "prediction_set_size": s["prediction_set_size"],
            "predicted_label": s["predicted_class"], "model_version": version,
        })
        counts[s["predicted_class"]] += 1
        areas[s["predicted_class"]] += s["mask_area_px"]
        if s["prediction_set_size"] > 1:
            review_rows.append({
                "request_id": request_id, "image_id": image_url, "segment_id": s["segment_id"],
                "reef_location_id": reef_id, "image_url": image_url,
                "prediction_set": s["prediction_set"],
                "conf_healthy": cs.get("healthy", 0.0), "conf_bleached": cs.get("bleached", 0.0),
                "model_version": version, "status": "pending",
            })
    supabase.log_segments(inf_rows)
    supabase.log_review(review_rows)
    supabase.log_snapshot({
        "reef_location_id": reef_id, "request_id": request_id, "source_image_id": image_url,
        "total_segments": summary["total_segments"],
        "healthy_count": counts["healthy"], "bleached_count": counts["bleached"],
        "total_area_px": total_area,
        "healthy_area_px": areas["healthy"], "bleached_area_px": areas["bleached"],
    })
