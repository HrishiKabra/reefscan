// Frozen inference response contract (2-class). Mirrors CLAUDE.md / the Phase 5 backend.
// Frontend builds entirely against this; Phase 5 is a one-line swap in lib/api.ts.

export type CoralClass = "healthy" | "bleached";
export type JobStatus = "queued" | "processing" | "complete" | "failed";

export interface Segment {
  segment_id: number;
  mask_area_px: number;
  bbox: [number, number, number, number]; // [x0, y0, x1, y1] in source-image px
  predicted_class: CoralClass;
  prediction_set: CoralClass[];
  prediction_set_size: number; // 1 = confident, >1 = uncertain -> review_queue
  confidence_scores: Record<CoralClass, number>; // raw softmax
  coverage_pct: number;
}

export interface InferenceSummary {
  total_segments: number;
  area_weighted: { healthy_pct: number; bleached_pct: number };
  uncertain_segments: number;
  dominant_status: CoralClass;
}

export interface InferenceResponse {
  job_id: string;
  status: JobStatus;
  processing_time_ms: number;
  image_url: string;
  model_version: string;
  // assumed natural image size for the overlay coordinate space (added client-side if absent)
  image_width?: number;
  image_height?: number;
  segments: Segment[];
  summary: InferenceSummary;
  error_message?: string | null;
}

// ----- admin review queue -----
export interface ReviewItem {
  id: string;
  image_id: string;
  segment_id: number;
  patch_url: string;
  image_url: string;
  prediction_set: CoralClass[];
  confidence_scores: Record<CoralClass, number>;
  model_version: string;
  reef_location: string;
  created_at: string; // ISO
  status: "pending" | "confirmed" | "rejected";
}

// ----- temporal tracker -----
export interface ReefLocation {
  id: string;
  name: string;
  lat: number;
  lng: number;
}

export interface HealthSnapshot {
  date: string; // ISO date
  healthy_pct: number;
  bleached_pct: number;
  avg_set_size?: number | null; // drift proxy (mock only; real drift lives in /observability)
  total_segments: number;
}

// ----- observability (Phase 7) -----
export interface DriftPoint { date: string; avg_set_size: number; n: number }
export interface LatencyPoint { date: string; p50: number; p95: number; n: number }
export interface ClassDistribution {
  current: Record<string, number>;
  baseline: Record<string, number>;
  current_window: [string, string] | null;
  baseline_window: [string, string] | null;
}
export interface Observability {
  drift: DriftPoint[];
  latency: LatencyPoint[];
  class_distribution: ClassDistribution;
  total_logs: number;
}

export const CLASS_META: Record<CoralClass, { label: string; varName: string; short: string }> = {
  healthy: { label: "Healthy", varName: "var(--healthy)", short: "HLT" },
  bleached: { label: "Bleached", varName: "var(--bleached)", short: "BLC" },
};
