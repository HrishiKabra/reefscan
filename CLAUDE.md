# CLAUDE.md — ReefScan

This file is the single source of truth for any Claude Code session working on this
repo. Read it fully before doing anything. It encodes every hard constraint and every
architectural decision already made, so you do not need to be re-briefed.

If a request conflicts with something here, stop and ask — do not silently deviate.

---

## What ReefScan is

End-to-end underwater image analysis pipeline. Users upload underwater photos or short
video clips. The system:

1. Enhances the image (WaterNet)
2. Extracts frames if video (PySceneDetect), passthrough if image
3. Segments coral structures (SAM2, **auto-prompted via Automatic Mask Generator — no
   manual clicks, no trained detection head**)
4. Classifies health per segment (DINOv2-B backbone + linear/fine-tuned head)
5. Wraps classification in conformal prediction → prediction *sets* with a coverage
   guarantee, not bare point labels
6. Logs every inference to Supabase
7. Displays results on a Next.js dashboard with a temporal tracker per reef location

It is a **portfolio project** for ML/MLOps interviews (NVIDIA, AI labs, MLOps shops)
and will later be the specialist baseline in a VLM benchmark paper. Therefore: clean
eval harness, clean model versioning, real MLOps thinking (observability, uncertainty
quantification, data flywheel), defensible model choices over tutorial defaults.

---

## Hard constraints — DO NOT violate without asking

### Compute / cost
- **No paid GPU at build time.** Training runs on **Kaggle free tier** (P100, 9-hour
  session max, 30 hrs/week).
- **Inference runs on Hugging Face Spaces free CPU** (16 GB RAM, 2 vCPU). *(This
  replaces the brief's original "Render free tier" — Render free is 512 MB RAM and
  cannot hold SAM2 + DINOv2-B + WaterNet. Decision locked in Phase 0.)*
- Every training script MUST:
  1. Checkpoint every epoch to `/kaggle/working/checkpoints/` and copy to a Kaggle
     output Dataset (so it survives session death and is resumable)
  2. Load from the latest checkpoint if one exists (resume, don't restart)
  3. Log to Weights & Biases throughout
  4. Complete within a single 9-hour Kaggle session
- Trained weights get pushed to **Hugging Face Hub** and loaded from there at inference.

### Free-tier gotchas to remember
- **Supabase free tier auto-pauses after ~1 week of inactivity.** A cold dashboard is
  not a bug. This MUST be noted in the README and demo notes.
- HF Spaces free CPU has cold starts. Acceptable; design for it.
- Cloudflare R2: zero egress, used for image/video storage.
- W&B: free academic tier.

### Model choices (do not swap without asking)
- **Segmentation:** SAM2, model size **SAM2-Hiera-Small** (CPU inference budget).
  Prompted via the **Automatic Mask Generator** (grid of point prompts). A trained
  detection head is explicitly **future work** — documented as such in the README.
  **AMG settings locked by the Phase 1.5 sweep (see below): `points_per_side=16`, AMG
  input downscaled so the longest edge = 512 px.** These are the canonical values — do
  NOT hardcode different ones in segmenter.py; read them from config/constants.
- **Classification backbone:** DINOv2-B (`facebook/dinov2-base`). Linear probe first;
  full fine-tune (unfreeze last 2 transformer blocks + head) as an ablation.
- **Uncertainty:** conformal prediction via **MAPIE 1.x**, `SplitConformalClassifier`
  with LAC/APS scoring. Split conformal on a held-out calibration set. Coverage target
  **90%**.
- **Enhancement:** WaterNet (pretrained, inference only). No CLAHE substitute — the
  WaterNet-over-CLAHE rationale is a README talking point.
- **Frame extraction:** PySceneDetect (scene-change based, NOT fixed-fps).

### Classification labels — INITIAL MODEL IS 2-CLASS
```
healthy | bleached          <- modeled now (model head, UI, conformal sets)
dead | algae_covered        <- RESERVED in the DB enum, NOT modeled yet
```
Locked from Phase 2 EDA of the NOAA dataset (see Dataset below), which contains only two
health states (`CORAL`=healthy, `CORAL_BL`=bleached) — no taxonomy, no dead/algae.

**Initial model is 2-class (healthy/bleached). `dead` and `algae_covered` enum values are
reserved for future extension via ReefNet supplementation — no migration needed.** The DB
`coral_label` enum keeps all 4 values (Postgres enum changes are painful, unused values are
harmless); the model/UI/conformal run 2-class. `LABEL_MAPPING` (`backend/data/label_mapping.py`)
and `CLASSES` are the single source of truth for the active class set.

---

## Key architectural decisions (Phase 0, locked)

### The train→inference bridge (read this — it's the subtle one, and now a CLEANER story)
The NOAA training data is **whole-image colony patches** (one health label per pre-cropped
image), and the pipeline classifies **SAM2 segments** at inference. Reconciled deliberately:

- **Training:** DINOv2 is trained on the **whole colony patch** (resize → 224×224,
  ImageNet-normalize). No point/centroid crop — the images are already colony-level crops.
- **Inference:** each SAM2 mask is classified by cropping the **mask's bounding box**, then
  the same resize-224 + normalize tail.
- **Patch geometry (must match train + inference exactly):** resize to **224×224**,
  ImageNet stats. At inference the crop is the mask bbox; the mask shape itself is ignored
  for the crop.
- **Why this is a better story than the original centroid-patch plan (call this out in the
  README):** the training distribution (colony-level crops) closely matches what SAM2 mask
  bboxes produce at inference, so train/inference domains are aligned by construction —
  tighter than training on point-centered crops would have been.
- **Why segment at all?** Segmentation provides spatial display and per-class coverage
  estimation; classification operates on the mask-bbox crops. The architectural split is
  intentional.
- *(For a future point-annotated source, the centroid-crop path still exists:
  `transforms.crop_patch_around_point` + `backend.data.split`. Not used for NOAA.)*

### Data splitting
- **Use the NOAA dataset's NATIVE train/val/test splits** (the imagefolder `train|val|test`
  dirs). They are site/year-controlled to prevent leakage — do NOT re-split this dataset.
- `backend/data/split.py` (by-image stratified split) is **retained for future
  point-annotated sources only**, not used for NOAA.

### Dataset (current)
- **NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset** (Hugging Face). ImageFolder format:
  `<split>/<RAW_LABEL>/<image>.PNG`, raw labels `CORAL` / `CORAL_BL`. ~10,419 pre-cropped
  colony patches, image-level labels, native split train 7,292 / val 1,562 / test 1,565,
  ~62% healthy / 38% bleached (imbalance 1.7:1 → standard cross-entropy, no resampling).
- Loader: `backend.data.patch_dataset.CoralPatchDataset.from_imagefolder(root, split)`.
- The original brief assumed CoralNet point-annotation CSV (`image_name,label,row,col`).
  That path (point crops + stratified split) is retained for a future source but is NOT
  what we train on now.

### Health metric storage
- `health_snapshots` stores **both** per-class segment **counts** and per-class **pixel
  areas**, so the tracker can render count-% or area-% with no migration. Area-% is the
  featured metric in the README.

### Reef location
- `reef_locations` is its own table. Uploads attach via `reef_location_id` chosen at
  upload time. EXIF-GPS auto-prefill is future work.

---

## Phase 1.5 — Feasibility spike findings (LOCKED, measured)

Measured on Apple Silicon CPU with real `facebook/dinov2-base` + real
`facebook/sam2-hiera-small`, WaterNet stubbed as identity, fp32. (`spike/feasibility_spike.py`,
`spike/amg_sweep.py`.) HF Spaces free is x86 2-vCPU — slower — so treat these as a floor.

**Conclusion 1 — RAM is cleared. No lazy loading.**
Peak RSS with BOTH models resident was **~0.9 GB** (DINOv2-B ≈ +0.44 GB, SAM2-Hiera-S
brings it to +0.73 GB over a 0.19 GB baseline). On the 16 GB Spaces box, even with the
FastAPI process + activation overhead this stays a few GB. Therefore: **load both models
once at startup and keep them resident permanently. Do NOT release SAM2 between calls.
Do NOT lazy-load.** The earlier lazy-load contingency is dropped.

**Conclusion 2 — SAM2 AMG is the latency bottleneck, not DINOv2.**
DINOv2-B forward = **166 ms / 224 patch** (≈ 4.5 s for ~27 masks — negligible). SAM2 AMG
at the old default `points_per_side=32` = **~25 s / 512 px image** here, i.e. ~1–2 min on
Spaces. **If we ever do the optional ONNX/quantization optimization (Phase E "Path B"),
the target is the SAM2 image encoder specifically — not DINOv2.**

**Conclusion 3 — AMG setting locked: `points_per_side=16`, input longest-edge 512 px.**
Sweep (this machine):

| size | points_per_side | prompts | time (s) | masks |
|-----:|----------------:|--------:|---------:|------:|
| 384  | 8  | 64   | 2.7  | 5  |
| 384  | 16 | 256  | 6.3  | 17 |
| 384  | 32 | 1024 | 22.6 | 24 |
| 512  | 8  | 64   | 2.2  | 5  |
| 512  | 16 | 256  | 6.4  | 19 |
| 512  | 32 | 1024 | 25.4 | 27 |

Chosen **pps=16 @ 512 px → ~6.4 s, ~19 masks** here. pps=8 is too coarse (5 masks); pps=32
quadruples prompts for ~40% more masks (diminishing returns). pps=16 leaves ~4× headroom
so the uncertain Spaces x86 penalty (→ ~15–25 s) stays well inside the async budget.
**This is the canonical AMG config. segmenter.py MUST read it from a named constant/config,
never inline a different magic number.**

## Active learning loop (data flywheel)
- A prediction is "uncertain" when its conformal **`prediction_set_size > 1`** (model
  can't commit to one class at 90% coverage).
- Uncertain predictions are written to the `review_queue` table.
- `/admin/review` shows flagged images + the candidate labels; a human confirms.
- Confirmed labels are written to `human_labels`.
- **Retraining is manual**, triggered only when `human_labels` reaches **100 new rows**
  (to conserve Kaggle GPU budget). Never wire automatic retraining.

## Observability
- Every classified segment logs a row to `inference_logs`: `image_id`, timestamp,
  `latency_ms`, per-class raw softmax, `prediction_set`, `prediction_set_size`,
  `model_version` (+ `reef_location_id`, `request_id`, `segment_id`).
- Dashboard (Phase 7) computes everything from raw Supabase SQL — no external
  observability tool:
  - Rolling average of `prediction_set_size` over time → drift proxy (rising = shift)
  - Inference latency p50 / p95 over time
  - Class distribution: current week vs baseline week

---

## Tech stack
- **Backend:** FastAPI (async), Python 3.11+
- **Frontend:** Next.js 14, TypeScript, Tailwind CSS
- **DB:** Supabase (Postgres), free tier
- **Storage:** Supabase Storage (public bucket `reefscan-uploads`, via the service_role
  client — no extra account). Cloudflare R2 still supported as an override (set all R2_* env).
- **Deploy:** Vercel (frontend), HF Spaces free CPU (backend)
- **Experiment tracking:** W&B free academic tier
- **Training:** Kaggle Notebooks (P100)

## Repo layout
```
reefscan/
├── CLAUDE.md                  # this file
├── README.md                  # portfolio writeup — explains every decision
├── backend/
│   ├── main.py                # FastAPI app (Phase 5)
│   ├── inference/
│   │   ├── pipeline.py        # orchestrates the full pipeline (Phase 5)
│   │   ├── segmenter.py       # SAM2 AMG wrapper (Phase 5)
│   │   ├── classifier.py      # DINOv2 + conformal (Phase 4/5)
│   │   ├── enhancer.py        # WaterNet (Phase 5)
│   │   └── frame_extractor.py # PySceneDetect (Phase 5)
│   ├── models/
│   │   └── train_dinov2.py    # training entrypoint (Phase 3)
│   ├── data/                  # Phase 2 (DONE)
│   │   ├── label_mapping.py   #   CLASSES + LABEL_MAPPING (2-class, locked)
│   │   ├── transforms.py      #   resize-224 + ImageNet norm
│   │   ├── patch_dataset.py   #   CoralPatchDataset (ImageFolder loader) — USED
│   │   ├── split.py           #   by-image stratified split — RESERVED for future
│   │   └── eda.py             #   raw-label / imbalance / loss-rec report
│   ├── db/
│   │   └── schema.sql         # Supabase schema (Phase 1, DONE)
│   └── requirements.txt
├── frontend/
│   ├── app/{page.tsx, admin/review/, tracker/}
│   └── package.json
├── notebooks/
│   └── 01_train_dinov2.ipynb   # one-shot training+conformal (Colab/Kaggle, Phase 3/4)
└── data/
    └── annotations_noaa.csv   # generated NOAA imagefolder index (image_name,label,split)
```

---

## Build phases & status
- [x] **Phase 0** — Q&A, decisions locked
- [x] **Phase 1** — scaffold, this CLAUDE.md, `schema.sql`, `requirements.txt`
- [x] **Phase 1.5** — feasibility/latency spike DONE. RAM cleared, AMG setting locked,
      Phase 5 forced async. See "Phase 1.5 — Feasibility spike findings" above.
- [x] **Phase 2** — DONE. `CoralPatchDataset` (ImageFolder), transforms, EDA; 2-class
      mapping locked (`CORAL`→healthy, `CORAL_BL`→bleached) from NOAA EDA; native splits.
- [~] **Phase 3** — linear probe DONE (Colab, 10 ep): **test acc 0.857, macro-F1 0.846**.
      Weights + config + conformal CONFIRMED on HF `HrishiKabra/reefscan-dinov2-coral`
      under `linear_probe/`. W&B logged. Full fine-tune (~4 hr) still TODO as the ablation.
      `model_version=reefscan-dinov2-coral-v1-linearprobe`.
- [x] **Phase 4** — DONE. Split-conformal **LAC** at 90% target, calibrated on the full
      val split (1,562 images). **qhat=0.628, realized coverage 0.914 on test, avg set
      1.12, ~12% uncertain.** `linear_probe/conformal.json` on the Hub; backend loads it.
      **DEVIATION from the locked "MAPIE 1.x" decision:** implemented hand-rolled LAC
      (≈15 lines, transparent, no MAPIE-API friction). Flag for the user — keep, or
      re-do via MAPIE for the "I used MAPIE" talking point. Backend conformal-set unit
      tests live in `backend/tests/` (API contract); add a dedicated coverage test if kept.
- [~] **Phase 5** — SCAFFOLD DONE, contract-verified in stub mode. FastAPI **async job**
      pipeline: `POST /infer` (multipart file or `url`) → `202 {job_id}`; `GET /infer/{job_id}`
      polls → `InferenceResponse`; `GET /health`. Images+video share one path. Background
      worker runs the pipeline in a threadpool. Real SAM2 (AMG pps=16/512) + DINOv2 + LAC
      conformal load from HF `{stage}/model.safetensors` + `{stage}/conformal.json` at
      startup; **falls back to STUB mode** (contract-valid synthetic output) when weights
      absent (`REEFSCAN_STUB=1` forces it). Supabase (`inference_logs`/`review_queue`/
      `health_snapshots`/`jobs`) + R2 wired with **no-op fallbacks** when creds absent.
      Tests: `backend/tests/` (10 pass). Real classifier path VERIFIED loading the trained
      weights from the Hub. Read endpoints (`/review-queue` + confirm, `/reef-locations`,
      `/snapshots`, `/observability`) added, returning frontend-shaped JSON. Frontend
      `lib/api.ts` fully wired: env `NEXT_PUBLIC_REEFSCAN_API` → real backend, else mock.
      Deploy: `deploy/Dockerfile` + `deploy/README_SPACE.md` (HF Docker Space),
      `backend/db/seed_demo.sql` (populated demo). **Supabase LIVE** — project
      `jscowijwqdmulginnmow` (`reefscan`): schema + seed applied, RLS enabled on all tables
      (backend uses the **service_role** key via `SUPABASE_KEY`; anon denied), backend reads
      verified against it (3 reefs / 252 logs / realistic drift). **Storage:** Supabase Storage
      bucket `reefscan-uploads` (public) — upload+public-read verified; no Cloudflare needed.
      **HF Space:** `HrishiKabra/reefscan-api` (Docker) created + pushed + secrets set
      (`SUPABASE_URL`/`SUPABASE_KEY`/`HF_MODEL_STAGE=finetune`); serves the **fine-tune**
      model (test acc 0.895 / F1 0.887). **LIVE + verified end-to-end** (real image → 18
      SAM2 segments → DINOv2 fine-tune → conformal → stored in Supabase Storage → logged).
      **GitHub:** github.com/HrishiKabra/reefscan (public). **Remaining (user action):**
      deploy `frontend/` to Vercel with
      `NEXT_PUBLIC_REEFSCAN_API=https://hrishikabra-reefscan-api.hf.space`.
      NOTE: `schema.sql` had a table-order bug (jobs before reef_locations) — FIXED.
      `seed_demo.sql` had a lateral-`random()` hoisting bug — FIXED via MATERIALIZED CTE.
      Supabase writes (httpx client shared across event-loop + executor threads) intermittently
      dropped `review_queue`/job-complete writes on the deployed box — FIXED with a
      serialize-lock + 3× retry (`persistence._sb_call`) + persist `result_json`/`completed_at`.
- [x] **Phase 6** — DONE (against mock). Next.js 14 + TS + Tailwind: analyze+overlay,
      admin review queue, temporal tracker. Builds clean, all 3 pages render-verified.
      **Phase 5 swap point = `frontend/lib/api.ts`** (replace mock returns with fetch()).
- [x] **Phase 7** — DONE (against mock + live `/observability`). `backend/observability.py`
      aggregates `inference_logs` → drift (rolling x̄ set size), latency p50/p95, class
      distribution (this week vs baseline). Frontend `app/dashboard/` renders all three.
      Build clean, render-verified.

**Linear probe unblocks Phases 4–6** — do not wait on the 4-hr fine-tune to start them.

- [x] **Phase 8** — portfolio hardening. Eval harness (`backend/eval.py`): finetune test
      **acc 0.895 / macro-F1 0.887 / ECE 0.046**; conformal **LAC** (cov 0.923, set 1.075;
      class-conditional reveals minority **bleached under-coverage 0.876** — marginal ≠
      per-class) **vs APS** (cov 0.996, set 1.95 — over-covers). **VLM benchmark**
      (`vlm_benchmark.py`): the specialist **beats zero-shot GPT-4o** (0.895/0.887/0.046 vs
      0.805/0.790/0.152) on the full test set (~$0.68). **Profiling** (`bench.py`): SAM2 AMG
      ~16.7s vs DINOv2 112ms/patch → SAM2 is ~88-99% of latency (the ONNX/quant target);
      naive int8 quant of DINOv2 REGRESSED latency 2.4× on ARM (honest negative). Also:
      CI (`.github/workflows/ci.yml`), `docker-compose.yml`, `retrain_trigger.py` (closes
      active-learning loop, verified live), HF **model card** pushed, README "Evaluation &
      benchmarks" section, `docs/eval/` artifacts (plots + json). TODO: demo GIF.

---

## Secrets / credentials (provided when their phase is reached; never hardcode)
- Supabase project URL + anon key → Phase 5
- **HF weights repo → `HrishiKabra/reefscan-dinov2-coral`** (Phase 5 loads weights from here;
  Phase 3 notebook pushes to it). Provided 2026-06-03.
- Cloudflare R2 bucket + keys → Phase 5
- W&B API key → Phase 3
Store all in env / `.env` (gitignored). Never commit secrets.

## Inference response contract (frozen — frontend Phase 6 builds against this)
`GET /infer/{job_id}` returns this shape (2-class). Phase 6 mocks it; Phase 5 is a 1-line swap.
```json
{ "job_id","status","processing_time_ms","image_url","model_version",
  "segments": [ { "segment_id","mask_area_px","bbox":[x0,y0,x1,y1],"predicted_class",
                  "prediction_set":["..."],"prediction_set_size","confidence_scores":{"healthy","bleached"},
                  "coverage_pct" } ],
  "summary": { "total_segments","area_weighted":{"healthy_pct","bleached_pct"},
               "uncertain_segments","dominant_status" } }
```
`prediction_set_size==1` → confident (single label). `>1` → uncertain: UI flags it + it goes to
`review_queue`.

## Open questions still to resolve (do not assume)
- ~~Phase 2 label mapping~~ — RESOLVED: 2-class (`CORAL`→healthy, `CORAL_BL`→bleached).
- ~~Phase 4 calibration unit~~ — RESOLVED: calibrated on the full **val split (1,562
  images)** held out from training; realized test coverage 0.914.
- ~~Phase 5 video handling~~ — RESOLVED by Phase 1.5: single async job pipeline for both
  images and video.

## Working conventions
- Don't pre-implement a later phase's logic while scaffolding an earlier one. Stubs carry
  a docstring naming their phase.
- Keep the 4-label vocabulary identical across DB enum, model head, and UI.
- Any model version string used at inference must be logged to `inference_logs`.
