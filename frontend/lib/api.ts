// API surface — talks to the real FastAPI backend when NEXT_PUBLIC_REEFSCAN_API is set,
// otherwise resolves from lib/mock.ts (dev / no-backend). This is the Phase 5 swap point:
// the backend exposes exactly these routes (see backend/main.py).
//
//   POST /infer                       -> { job_id }            (submitJob)
//   GET  /infer/{job_id}              -> InferenceResponse      (getInferenceResult, polled)
//   GET  /review-queue                -> ReviewItem[]
//   POST /review-queue/{id}/confirm   -> { ok }                (confirmLabel)
//   GET  /reef-locations              -> ReefLocation[]
//   GET  /reef-locations/{id}/snapshots -> HealthSnapshot[]
//   GET  /observability               -> Observability

import {
  labelsConfirmedThisCycle, mockInference, mockLoadTest, mockObservability, mockReefLocations,
  mockReviewQueue, mockSnapshots, RETRAIN_THRESHOLD,
} from "./mock";
import type {
  HealthSnapshot, InferenceResponse, LoadTest, Observability, ReefLocation, ReviewItem,
} from "./types";

const API = process.env.NEXT_PUBLIC_REEFSCAN_API?.replace(/\/$/, "") || "";
const live = !!API;
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

/** POST /infer — enqueue a job, returns its id immediately. */
export async function submitJob(input: File | string): Promise<{ job_id: string }> {
  if (!live) { await wait(450); return { job_id: mockInference.job_id }; }
  const form = new FormData();
  if (typeof input === "string") form.append("url", input);
  else form.append("file", input);
  const r = await fetch(`${API}/infer`, { method: "POST", body: form });
  if (!r.ok) throw new Error(`infer submit -> ${r.status}`);
  return r.json();
}

/** GET /infer/{job_id} — poll until the job completes (or fails / times out). */
export async function getInferenceResult(jobId: string): Promise<InferenceResponse> {
  if (!live) { await wait(1600); return { ...mockInference, job_id: jobId }; }
  const deadline = Date.now() + 4 * 60_000;
  while (Date.now() < deadline) {
    const data = await get<InferenceResponse>(`/infer/${jobId}`);
    if (data.status === "complete") return data;
    if (data.status === "failed") throw new Error(data.error_message || "inference failed");
    await wait(2500);
  }
  throw new Error("inference timed out");
}

export async function getReviewQueue(): Promise<ReviewItem[]> {
  if (!live) { await wait(300); return mockReviewQueue; }
  return get<ReviewItem[]>("/review-queue");
}

export async function confirmLabel(
  itemId: string, label: "healthy" | "bleached",
): Promise<{ ok: boolean; itemId: string; label: string }> {
  if (!live) { await wait(250); return { ok: true, itemId, label }; }
  const r = await fetch(`${API}/review-queue/${itemId}/confirm`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  const j = await r.json();
  return { ok: !!j.ok, itemId, label };
}

export async function getReefLocations(): Promise<ReefLocation[]> {
  if (!live) { await wait(150); return mockReefLocations; }
  return get<ReefLocation[]>("/reef-locations");
}

export async function getSnapshots(reefId: string): Promise<HealthSnapshot[]> {
  if (!live) { await wait(300); return mockSnapshots[reefId] ?? []; }
  return get<HealthSnapshot[]>(`/reef-locations/${reefId}/snapshots`);
}

export async function getObservability(): Promise<Observability> {
  if (!live) { await wait(300); return mockObservability; }
  return get<Observability>("/observability");
}

/** GET /loadtest — the committed serving load-test artifact (null if none on the backend). */
export async function getLoadTest(): Promise<LoadTest | null> {
  if (!live) { await wait(200); return mockLoadTest; }
  try { return await get<LoadTest>("/loadtest"); } catch { return null; }
}

export { RETRAIN_THRESHOLD, labelsConfirmedThisCycle };
