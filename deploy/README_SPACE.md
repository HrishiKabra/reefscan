---
title: ReefScan API
emoji: 🪸
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ReefScan backend (FastAPI, async-job inference)

CPU inference API for coral-health analysis: SAM2 segmentation → DINOv2 classification →
conformal prediction sets. Loads weights + calibration from
[`HrishiKabra/reefscan-dinov2-coral`](https://huggingface.co/HrishiKabra/reefscan-dinov2-coral).

## Endpoints
- `POST /infer` — multipart `file` (image/video) or form `url` → `202 {job_id}`
- `GET /infer/{job_id}` — poll → `InferenceResponse` (status + results)
- `GET /health` — model load status + last latency
- `GET /review-queue`, `POST /review-queue/{id}/confirm`
- `GET /reef-locations`, `GET /reef-locations/{id}/snapshots`
- `GET /observability`

## Space secrets (Settings → Variables and secrets)
| name | required | purpose |
|---|---|---|
| `SUPABASE_URL` / `SUPABASE_ANON_KEY` | for logging | inference_logs / review_queue / snapshots / jobs |
| `R2_ENDPOINT` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET` | for image display | store uploaded images |
| `HF_MODEL_STAGE` | optional | `linear_probe` (default) or `finetune` |
| `HF_TOKEN` | only if model repo is private | weights download |
| `REEFSCAN_STUB` | optional | `1` forces synthetic inference (no models) |

Without Supabase/R2 the API still runs: logging is a no-op and image urls are placeholders.
First request is slow (cold start downloads ~0.7 GB of weights); SAM2 AMG is ~15–25 s/image
on the free 2-vCPU box (Phase 1.5), which is why inference is an async job.
