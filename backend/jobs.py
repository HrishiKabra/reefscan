"""Async job store + background worker. Phase 5.

POST /infer creates a job (status 'queued') and schedules run_job; GET /infer/{job_id}
reads back status + results. In-memory store is the source of truth for polling; the
Supabase `jobs` table is mirrored best-effort (single-process worker on HF Spaces).
The CPU-bound pipeline runs in a threadpool so the event loop stays responsive.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from .inference import pipeline
from .persistence import new_request_id, supabase
from .schemas import InferenceResponse

logger = logging.getLogger(__name__)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self, kind: str, reef_location_id: Optional[str], source_url: str = "") -> str:
        job_id = new_request_id()
        self._jobs[job_id] = {"status": "queued", "result": None, "error": None,
                              "kind": kind, "reef": reef_location_id}
        supabase.upsert_job({"job_id": job_id, "status": "queued", "source_kind": kind,
                             "source_url": source_url, "reef_location_id": reef_location_id})
        return job_id

    def update(self, job_id: str, **kw: Any) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(kw)
            status = self._jobs[job_id]["status"]
            row: dict[str, Any] = {"job_id": job_id, "status": status}
            if kw.get("error"):
                row["error_message"] = kw["error"]
            if kw.get("result"):
                row["result_json"] = kw["result"]
            if status in ("complete", "failed"):
                from datetime import datetime, timezone
                row["completed_at"] = datetime.now(timezone.utc).isoformat()
            supabase.upsert_job(row)

    def response(self, job_id: str) -> Optional[InferenceResponse]:
        j = self._jobs.get(job_id)
        if j is None:
            return None
        result = j.get("result") or {}
        return InferenceResponse(job_id=job_id, status=j["status"],
                                 error_message=j.get("error"), **result)


async def run_job(store: JobStore, job_id: str, data: bytes, kind: str,
                  reef_location_id: Optional[str]) -> None:
    store.update(job_id, status="processing")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, pipeline.run, job_id, data, kind, reef_location_id)
        store.update(job_id, status="complete", result=result)
    except Exception as e:  # noqa: BLE001
        logger.exception("job %s failed", job_id)
        store.update(job_id, status="failed", error=str(e))
