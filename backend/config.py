"""Runtime configuration from environment. Phase 5.

All secrets come from env / .env (never hardcoded — see CLAUDE.md). Everything degrades
gracefully when a secret is absent so the app runs locally without Supabase / R2 / weights:
  - no Supabase creds  -> logging is a no-op
  - no R2 creds        -> uploads return a local placeholder url
  - weights not on Hub -> inference runs in STUB mode (contract-valid synthetic output)
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # --- model weights (HF Hub) ---
    hf_repo: str = os.getenv("HF_MODEL_REPO", "HrishiKabra/reefscan-dinov2-coral")
    hf_stage: str = os.getenv("HF_MODEL_STAGE", "linear_probe")
    hf_token: str | None = os.getenv("HF_TOKEN") or None

    # --- Supabase (logging) ---
    # Use the SERVICE_ROLE key here (server-side only) — tables have RLS enabled, so the
    # service_role key (which bypasses RLS) is required to read/write. SUPABASE_KEY is the
    # canonical name; SUPABASE_ANON_KEY is accepted for backward-compat.
    supabase_url: str | None = os.getenv("SUPABASE_URL") or None
    supabase_key: str | None = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY") or None

    # --- object storage ---
    # Default = Supabase Storage (reuses the Supabase service_role client; no extra account).
    # Optionally override with Cloudflare R2 by setting all four R2_* vars.
    storage_bucket: str = os.getenv("STORAGE_BUCKET", "reefscan-uploads")
    r2_endpoint: str | None = os.getenv("R2_ENDPOINT") or None
    r2_key_id: str | None = os.getenv("R2_ACCESS_KEY_ID") or None
    r2_secret: str | None = os.getenv("R2_SECRET_ACCESS_KEY") or None
    r2_bucket: str | None = os.getenv("R2_BUCKET") or None

    # --- locked AMG config (Phase 1.5; do NOT inline elsewhere) ---
    amg_points_per_side: int = 16
    amg_longest_edge: int = 512

    # --- conformal ---
    coverage_alpha: float = 0.10  # 90% target; qhat loaded from HF conformal.json

    # --- modeling ---
    classes: tuple[str, ...] = ("healthy", "bleached")
    input_size: int = 224

    # force synthetic inference (also auto-enabled if weights fail to load)
    stub_mode: bool = _flag("REEFSCAN_STUB")

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    @property
    def r2_enabled(self) -> bool:
        return bool(self.r2_endpoint and self.r2_key_id and self.r2_secret and self.r2_bucket)


settings = Settings()
