"""Persistence wrappers: Cloudflare R2 (objects) + Supabase (logging). Phase 5.

Both degrade to safe no-ops when their credentials are absent, so the pipeline runs
end-to-end locally. Tables match backend/db/schema.sql.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Optional

from .config import settings

logger = logging.getLogger(__name__)

# The supabase client is httpx-based and shared across the event-loop thread (job status
# updates) and the run_in_executor worker thread (pipeline logging). Serialize all access
# and retry transient transport errors so best-effort logging is actually reliable on the
# (slower) deployed box, where unsynchronized/stale connections intermittently dropped writes.
_SB_LOCK = threading.Lock()
_SB_RETRIES = 3


def _sb_call(fn: Callable[[], Any]) -> Any:
    last: Exception | None = None
    with _SB_LOCK:
        for i in range(_SB_RETRIES):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001  (transient transport/connection errors)
                last = e
                time.sleep(0.3 * (i + 1))
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Object storage — Supabase Storage by default (no extra account), R2 if configured.
# ---------------------------------------------------------------------------
class ObjectStore:
    def __init__(self, supabase_client=None) -> None:
        self._r2 = None
        self._sb = supabase_client          # reuse the Supabase service_role client
        self._bucket = settings.storage_bucket
        if settings.r2_enabled:
            try:
                import boto3  # type: ignore
                self._r2 = boto3.client(
                    "s3", endpoint_url=settings.r2_endpoint,
                    aws_access_key_id=settings.r2_key_id,
                    aws_secret_access_key=settings.r2_secret,
                )
                logger.info("object store: Cloudflare R2 (bucket=%s)", settings.r2_bucket)
            except Exception as e:  # noqa: BLE001
                logger.warning("R2 init failed: %s", e)
        elif self._sb is not None:
            logger.info("object store: Supabase Storage (bucket=%s)", self._bucket)

    def upload(self, key: str, data: bytes, content_type: str) -> str:
        """Store bytes, return a public url. Falls back to a local placeholder if neither
        backend is configured (the frontend then shows its gradient placeholder)."""
        # 1) Cloudflare R2
        if self._r2 is not None:
            self._r2.put_object(Bucket=settings.r2_bucket, Key=key, Body=data,
                                ContentType=content_type)
            ep = (settings.r2_endpoint or "").rstrip("/")
            return f"{ep}/{settings.r2_bucket}/{key}"
        # 2) Supabase Storage (via the service_role client; bucket must be public)
        if self._sb is not None:
            try:
                _sb_call(lambda: self._sb.storage.from_(self._bucket).upload(
                    key, data, {"content-type": content_type, "upsert": "true"}))
                url = self._sb.storage.from_(self._bucket).get_public_url(key)
                return url if isinstance(url, str) else key
            except Exception as e:  # noqa: BLE001
                logger.warning("supabase storage upload failed: %s", e)
        # 3) local placeholder
        return f"local://uploads/{key}"


# ---------------------------------------------------------------------------
# Supabase logging
# ---------------------------------------------------------------------------
class SupabaseLogger:
    def __init__(self) -> None:
        self._client = None
        if settings.supabase_enabled:
            try:
                from supabase import create_client  # type: ignore
                self._client = create_client(settings.supabase_url, settings.supabase_key)
                logger.info("Supabase logging enabled")
            except Exception as e:  # noqa: BLE001
                logger.warning("Supabase init failed, logging disabled: %s", e)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _insert(self, table: str, rows: list[dict[str, Any]]) -> None:
        if self._client is None or not rows:
            return
        try:
            _sb_call(lambda: self._client.table(table).insert(rows).execute())
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase insert into %s failed: %s", table, e)

    # ---- jobs ----
    def upsert_job(self, row: dict[str, Any]) -> None:
        if self._client is None:
            return
        try:
            _sb_call(lambda: self._client.table("jobs").upsert(row).execute())
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase upsert job failed: %s", e)

    # ---- inference_logs (one row per classified segment) ----
    def log_segments(self, rows: list[dict[str, Any]]) -> None:
        self._insert("inference_logs", rows)

    # ---- review_queue (uncertain predictions) ----
    def log_review(self, rows: list[dict[str, Any]]) -> None:
        self._insert("review_queue", rows)

    # ---- health_snapshots (per-upload aggregate) ----
    def log_snapshot(self, row: dict[str, Any]) -> None:
        self._insert("health_snapshots", [row])

    # ---- reads (Phase 6 review/tracker + Phase 7 dashboard) ----
    def _select(self, table: str, build) -> list[dict[str, Any]]:
        if self._client is None:
            return []
        try:
            return _sb_call(lambda: build(self._client.table(table).select("*")).execute().data) or []
        except Exception as e:  # noqa: BLE001
            logger.warning("supabase select from %s failed: %s", table, e)
            return []

    def review_queue(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._select("review_queue", lambda q: q.eq("status", "pending")
                            .order("created_at", desc=True).limit(limit))

    def reef_locations(self) -> list[dict[str, Any]]:
        return self._select("reef_locations", lambda q: q.order("name"))

    def snapshots(self, reef_id: str) -> list[dict[str, Any]]:
        return self._select("health_snapshots", lambda q: q.eq("reef_location_id", reef_id)
                            .order("snapshot_time"))

    def recent_logs(self, limit: int = 5000) -> list[dict[str, Any]]:
        return self._select("inference_logs", lambda q: q.order("ts", desc=True).limit(limit))

    def confirm_label(self, review_id: str, label: str, labeled_by: str = "admin") -> bool:
        if self._client is None:
            return False
        try:
            row = _sb_call(lambda: self._client.table("review_queue").select("*").eq("id", review_id).execute().data)
            if not row:
                return False
            r = row[0]
            _sb_call(lambda: self._client.table("human_labels").insert({
                "review_queue_id": review_id, "image_id": r.get("image_id"),
                "segment_id": r.get("segment_id"), "confirmed_label": label,
                "labeled_by": labeled_by,
            }).execute())
            _sb_call(lambda: self._client.table("review_queue").update({"status": "confirmed"}).eq("id", review_id).execute())
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("confirm_label failed: %s", e)
            return False


def new_request_id() -> str:
    return str(uuid.uuid4())


supabase = SupabaseLogger()
r2 = ObjectStore(supabase_client=supabase._client)  # exported as `r2` for pipeline import
