// Mock data for Phase 6 frontend development. Matches the frozen contract in types.ts.
// Phase 5 replaces lib/api.ts internals with real fetch() calls — these mocks stay for tests/storybook.

import type {
  HealthSnapshot, InferenceResponse, Observability, ReefLocation, ReviewItem, Segment,
} from "./types";

const NAT_W = 600;
const NAT_H = 380;

// Hand-authored segments within the NAT_W x NAT_H overlay space.
const rawSegments: Omit<Segment, "mask_area_px" | "coverage_pct">[] = [
  { segment_id: 1, bbox: [54, 38, 210, 172], predicted_class: "bleached",
    prediction_set: ["bleached"], prediction_set_size: 1,
    confidence_scores: { healthy: 0.08, bleached: 0.92 } },
  { segment_id: 2, bbox: [248, 56, 402, 198], predicted_class: "healthy",
    prediction_set: ["healthy"], prediction_set_size: 1,
    confidence_scores: { healthy: 0.94, bleached: 0.06 } },
  { segment_id: 3, bbox: [430, 40, 566, 176], predicted_class: "bleached",
    prediction_set: ["healthy", "bleached"], prediction_set_size: 2,
    confidence_scores: { healthy: 0.48, bleached: 0.52 } },
  { segment_id: 4, bbox: [40, 212, 182, 348], predicted_class: "healthy",
    prediction_set: ["healthy"], prediction_set_size: 1,
    confidence_scores: { healthy: 0.89, bleached: 0.11 } },
  { segment_id: 5, bbox: [212, 224, 332, 344], predicted_class: "bleached",
    prediction_set: ["bleached"], prediction_set_size: 1,
    confidence_scores: { healthy: 0.17, bleached: 0.83 } },
  { segment_id: 6, bbox: [360, 220, 474, 332], predicted_class: "healthy",
    prediction_set: ["healthy", "bleached"], prediction_set_size: 2,
    confidence_scores: { healthy: 0.55, bleached: 0.45 } },
  { segment_id: 7, bbox: [494, 206, 582, 322], predicted_class: "healthy",
    prediction_set: ["healthy"], prediction_set_size: 1,
    confidence_scores: { healthy: 0.78, bleached: 0.22 } },
];

function buildSegments(): Segment[] {
  const withArea = rawSegments.map((s) => {
    const [x0, y0, x1, y1] = s.bbox;
    const mask_area_px = Math.round((x1 - x0) * (y1 - y0) * 0.72);
    return { ...s, mask_area_px };
  });
  const totalArea = withArea.reduce((a, s) => a + s.mask_area_px, 0);
  return withArea.map((s) => ({
    ...s,
    coverage_pct: +((s.mask_area_px / totalArea) * 100).toFixed(1),
  }));
}

const segments = buildSegments();

function summarize(segs: Segment[]) {
  const total = segs.reduce((a, s) => a + s.mask_area_px, 0);
  const healthyArea = segs.filter((s) => s.predicted_class === "healthy").reduce((a, s) => a + s.mask_area_px, 0);
  const healthy_pct = +((healthyArea / total) * 100).toFixed(1);
  return {
    total_segments: segs.length,
    area_weighted: { healthy_pct, bleached_pct: +(100 - healthy_pct).toFixed(1) },
    uncertain_segments: segs.filter((s) => s.prediction_set_size > 1).length,
    dominant_status: (healthy_pct >= 50 ? "healthy" : "bleached") as "healthy" | "bleached",
  };
}

export const mockInference: InferenceResponse = {
  job_id: "8f1c2a90-4e3b-4d2a-9b77-0a1c2d3e4f55",
  status: "complete",
  processing_time_ms: 18400,
  image_url: "https://r2.example.com/uploads/FFS-B015_2019_transect_04.jpg",
  model_version: "reefscan-dinov2-coral-v1-linearprobe",
  image_width: NAT_W,
  image_height: NAT_H,
  segments,
  summary: summarize(segments),
};

// ----- review queue -----
export const RETRAIN_THRESHOLD = 100;
export const labelsConfirmedThisCycle = 73; // progress toward manual retrain trigger

export const mockReviewQueue: ReviewItem[] = [
  { id: "rq-101", image_id: "FFS-B015_2019_transect_04", segment_id: 3,
    patch_url: "", image_url: "", prediction_set: ["healthy", "bleached"],
    confidence_scores: { healthy: 0.48, bleached: 0.52 },
    model_version: "reefscan-dinov2-coral-v1-linearprobe",
    reef_location: "Kāneʻohe Bay — Patch Reef 12", created_at: "2026-05-31T14:22:00Z", status: "pending" },
  { id: "rq-102", image_id: "FFS-B015_2019_transect_04", segment_id: 6,
    patch_url: "", image_url: "", prediction_set: ["healthy", "bleached"],
    confidence_scores: { healthy: 0.55, bleached: 0.45 },
    model_version: "reefscan-dinov2-coral-v1-linearprobe",
    reef_location: "Kāneʻohe Bay — Patch Reef 12", created_at: "2026-05-31T14:22:00Z", status: "pending" },
  { id: "rq-103", image_id: "MOL-2402_2022_A_18", segment_id: 2,
    patch_url: "", image_url: "", prediction_set: ["healthy", "bleached"],
    confidence_scores: { healthy: 0.51, bleached: 0.49 },
    model_version: "reefscan-dinov2-coral-v1-linearprobe",
    reef_location: "Molokini Crater", created_at: "2026-05-30T09:05:00Z", status: "pending" },
  { id: "rq-104", image_id: "HAN-1097_2023_A_11", segment_id: 5,
    patch_url: "", image_url: "", prediction_set: ["healthy", "bleached"],
    confidence_scores: { healthy: 0.44, bleached: 0.56 },
    model_version: "reefscan-dinov2-coral-v1-linearprobe",
    reef_location: "Hanauma Bay", created_at: "2026-05-29T16:40:00Z", status: "pending" },
  { id: "rq-105", image_id: "HAN-1097_2023_A_22", segment_id: 9,
    patch_url: "", image_url: "", prediction_set: ["healthy", "bleached"],
    confidence_scores: { healthy: 0.59, bleached: 0.41 },
    model_version: "reefscan-dinov2-coral-v1-linearprobe",
    reef_location: "Hanauma Bay", created_at: "2026-05-29T16:41:00Z", status: "pending" },
];

// ----- temporal tracker -----
export const mockReefLocations: ReefLocation[] = [
  { id: "kaneohe", name: "Kāneʻohe Bay — Patch Reef 12", lat: 21.45, lng: -157.79 },
  { id: "molokini", name: "Molokini Crater", lat: 20.63, lng: -156.49 },
  { id: "hanauma", name: "Hanauma Bay", lat: 21.27, lng: -157.69 },
];

function series(base: number[], setSizes: number[]): HealthSnapshot[] {
  const dates = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"];
  return dates.map((d, i) => ({
    date: `${d}-01`,
    healthy_pct: base[i],
    bleached_pct: +(100 - base[i]).toFixed(1),
    avg_set_size: setSizes[i],
    total_segments: 14 + ((i * 3) % 9),
  }));
}

// ----- observability (Phase 7) -----
function _obsDays(n: number): string[] {
  // fixed demo dates ending 2026-06-27 (no Date.now in deterministic mock)
  const base = ["19", "20", "21", "22", "23", "24", "25", "26", "27"];
  return base.slice(-n).map((d) => `2026-06-${d}`);
}

export const mockObservability: Observability = {
  total_logs: 252,
  drift: _obsDays(9).map((date, i) => ({
    date, avg_set_size: +(1.06 + i * 0.03 + (i % 2 ? 0.01 : 0)).toFixed(3), n: 18 + (i % 4),
  })),
  latency: _obsDays(9).map((date, i) => ({
    date, p50: 16800 + i * 250, p95: 23200 + i * 400, n: 18 + (i % 4),
  })),
  class_distribution: {
    current: { healthy: 58.3, bleached: 41.7 },
    baseline: { healthy: 71.6, bleached: 28.4 },
    current_window: ["2026-06-21", "2026-06-27"],
    baseline_window: ["2026-06-14", "2026-06-20"],
  },
};

export const mockSnapshots: Record<string, HealthSnapshot[]> = {
  kaneohe: series(
    [82, 80, 77, 71, 66, 61, 58, 62, 61.2],
    [1.05, 1.06, 1.09, 1.14, 1.22, 1.31, 1.36, 1.28, 1.24],
  ),
  molokini: series(
    [91, 90, 89, 87, 85, 84, 82, 83, 84],
    [1.02, 1.03, 1.04, 1.05, 1.07, 1.08, 1.10, 1.09, 1.08],
  ),
  hanauma: series(
    [74, 70, 64, 58, 49, 44, 47, 51, 53],
    [1.10, 1.15, 1.24, 1.34, 1.46, 1.52, 1.44, 1.33, 1.29],
  ),
};
