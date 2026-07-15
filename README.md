# 🪸 ReefScan

**End-to-end coral reef health analysis with calibrated uncertainty — trained, deployed, and observable, entirely on free-tier infrastructure.**

Upload an underwater photo. ReefScan segments individual coral colonies, classifies each as **healthy** or **bleached**, and wraps every prediction in a **conformal prediction set** with a 90% coverage guarantee — so the model's uncertainty is *visible and actionable*, not hidden behind a single label. Uncertain colonies are routed to a human-review queue that feeds the next round of training (a data flywheel), and every inference is logged to power drift/latency/class-shift dashboards.

| | |
|---|---|
| 🔴 **Live demo** | **https://reefscan.vercel.app** |
| ⚙️ **Inference API** | https://hrishikabra-reefscan-api.hf.space ([`/health`](https://hrishikabra-reefscan-api.hf.space/health)) |
| 🤗 **Model + calibration** | [HrishiKabra/reefscan-dinov2-coral](https://huggingface.co/HrishiKabra/reefscan-dinov2-coral) |
| 📊 **Dataset** | [NOAA-PIFSC-ESD Coral Bleaching](https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset) |

**Try it:** on the [live demo](https://reefscan.vercel.app), hit **Run sample** for real inference on a reef photo, or drag in a test image from [`docs/sample-images/`](docs/sample-images) (a healthy reef + a real bleaching scene). The first run is slow (~45 s) — SAM2 on free CPU.

> ⚠️ **Demo note:** the Supabase free tier auto-pauses after ~1 week of inactivity and the Hugging Face Space cold-starts (it downloads ~0.7 GB of weights on first hit). A slow first load is expected, not a bug. Because SAM2 runs on a free 2-vCPU CPU, a single image takes ~45 s — which is exactly why inference is an **async job**, surfaced in the UI as a live "pipeline running" state.

---

## What it looks like

A tour through the four pages — analyze → dashboard → tracker → review:

![ReefScan demo](docs/screenshots/demo.gif)

**Analyze** — drop an image, get an annotated overlay (teal = healthy, amber = bleached, pulsing pink = *uncertain*), per-segment conformal sets, and an area-weighted health summary:

| Review queue (active learning) | Temporal tracker | Observability dashboard |
|---|---|---|
| ![Review](docs/screenshots/review.png) | ![Tracker](docs/screenshots/tracker.png) | ![Dashboard](docs/screenshots/dashboard.png) |

---

## Architecture

```
                 ┌─────────────────────── Hugging Face Space (FastAPI, free CPU) ───────────────────────┐
  image/video ──>│  POST /infer ─► async job  ─►  WaterNet enhance ─► PySceneDetect (video) / passthrough │
                 │                                 │                                                       │
   poll ◄────────│  GET /infer/{id}                ▼                                                       │
                 │                          SAM2 Hiera-S (Automatic Mask Generator, pps=16 @ 512px)        │
                 │                                 │   per mask: bbox crop ─► 224² ─► ImageNet norm         │
                 │                                 ▼                                                       │
                 │                          DINOv2-B classifier ─► softmax ─► split-conformal (LAC, 90%)    │
                 │                                 │                                                       │
                 │   Supabase (Postgres+Storage) ◄─┴─► inference_logs · review_queue · health_snapshots     │
                 └───────────────────────────────────────────────────────────────────────────────────────┘
                                                   ▲
   Next.js 14 (Vercel) ── Analyze · Review · Tracker · Dashboard ──────────┘   (one swap point: lib/api.ts)
```

**Stack:** FastAPI · Next.js 14 / TypeScript / Tailwind · Supabase (Postgres + Storage) · Hugging Face Spaces (Docker, free CPU) · Vercel · Weights & Biases · Kaggle/Colab for training.

---

## Results

Trained on the NOAA-PIFSC-ESD bleaching dataset (10,419 pre-cropped colony patches; native site/year-controlled train/val/test splits; ~62% healthy / 38% bleached). DINOv2-B backbone, evaluated on the held-out test split (1,565 images).

| stage | test accuracy | macro-F1 | conformal coverage* | avg. set size |
|---|---:|---:|---:|---:|
| Linear probe (frozen backbone, 1,538-param head) | 0.857 | 0.846 | 0.914 | 1.120 |
| **Full fine-tune** (last 2 blocks + head) | **0.895** | **0.887** | 0.923 | 1.075 |

\* Empirical coverage on the test split for a 90% target — the conformal guarantee holds (split conformal is designed to be slightly conservative). The fine-tune is more confident: average prediction-set size drops to **1.075**, i.e. ~92% of colonies get a single confident label and only ~8% are flagged uncertain.

The deployed Space serves the fine-tune model.

---

## Evaluation & benchmarks

Full harness in [`backend/eval.py`](backend/eval.py) (classifier + calibration + conformal),
[`backend/vlm_benchmark.py`](backend/vlm_benchmark.py) (vs. GPT-4o), and
[`backend/bench.py`](backend/bench.py) (latency profile). All numbers below are on the
held-out **test split (1,565 images)**.

### Specialist vs. a frontier VLM — *does an 86M-param specialist beat GPT-4o?*
Zero-shot GPT-4o (vision, forced binary choice + logprobs) on the **same** test set:

| model | accuracy | macro-F1 | ECE (calibration) |
|---|---:|---:|---:|
| GPT-4o (zero-shot) | 0.805 | 0.790 | 0.152 |
| **ReefScan specialist** (DINOv2-B, 86M) | **0.895** | **0.887** | **0.046** |

**Yes — by ~9 points accuracy, ~10 points macro-F1, and ~3× better calibrated.** GPT-4o
over-predicts "healthy" (bleached recall 0.71 vs the specialist's 0.84). The whole run cost
~$0.68. Takeaway: for a narrow, well-defined visual task with labeled data, a small fine-tuned
specialist still decisively beats a frontier generalist — and is far cheaper and more confident
to serve.

### Classification & calibration
Confusion matrix · reliability diagram (ECE = **0.046**, i.e. well-calibrated) · conformal coverage:

<p>
  <img src="docs/eval/confusion_matrix.png" width="32%" />
  <img src="docs/eval/reliability_diagram.png" width="32%" />
  <img src="docs/eval/conformal_coverage.png" width="33%" />
</p>

### Conformal: LAC vs APS, and a real calibration subtlety
| method | marginal coverage | avg. set size | healthy cov. | bleached cov. |
|---|---:|---:|---:|---:|
| **LAC** (shipped) | 0.923 | **1.075** | 0.952 | **0.876** |
| APS | 0.996 | 1.954 | 0.998 | 0.993 |

LAC hits the 90% target with tight, useful sets; APS over-covers (99.6%) with near-maximal
sets (1.95 ≈ "always both classes") — the wrong tool for binary. **The honest finding:** LAC's
*marginal* 92% coverage masks **class-conditional under-coverage of the minority bleached class
(87.6%)** — marginal conformal does not guarantee per-class coverage. Fix = Mondrian /
class-conditional conformal (future work). This is a real limitation, surfaced by measuring it.

### Inference profile — profile before you optimize
Coarse profile: **SAM2 AMG** dominates wall-clock; DINOv2 is ~112 ms/patch. The obvious next
move is "ONNX-quantize the SAM2 image encoder." **Profiling deeper showed that's the wrong
lever** ([`backend/optimize_sam2.py`](backend/optimize_sam2.py)):

| SAM2 AMG component | time | share |
|---|---:|---:|
| image encoder (`set_image`) | ~1.5 s | **~8%** |
| **mask decoding over the 256-point prompt grid** | ~17 s | **~92%** |

So the encoder is only ~8% — ONNX-ing it caps out there. Confirmed by measuring two encoder
optimizations that both **failed on CPU**: bf16 autocast (~14× *slower*) and `torch.compile`
(no gain). The real lever is the **prompt grid / decoder throughput**: `points_per_batch=128`
is a **measured free ~8% speedup with identical masks** (now shipped in `segmenter.py`; 256
regresses), and grid density (`points_per_side`) is the dominant knob but trades mask count
(swept in Phase 1.5). Separately, naive int8 quantization of DINOv2 **regressed** latency 2.4×
on ARM. Takeaway: every optimization here was *measured*, and the headline assumption was wrong
until it was profiled.

## Design decisions (the interesting part)

Every choice below was made deliberately and, where it mattered, **measured** before committing.

### 1. DINOv2-B over a supervised ViT
Self-supervised DINOv2 features transfer strongly with a *linear probe* — an 85.7% / 0.846-F1 baseline with a **1,538-parameter head** and a frozen backbone, before any fine-tuning. That's a clean, cheap, defensible baseline and a natural ablation against the full fine-tune.

### 2. SAM2 Automatic Mask Generator — no trained detection head
The pipeline auto-segments colonies with SAM2-Hiera-Small's **Automatic Mask Generator** (a grid of point prompts) — no manual clicks and, deliberately, **no trained detection head** (the dataset has image-level labels, not boxes, so there's nothing to train a detector on). A learned detector is documented future work, not a hidden dependency.

### 3. Conformal prediction (split conformal, LAC) — sets, not point labels
Instead of emitting a bare argmax, each colony gets a **prediction set** calibrated to cover the true class 90% of the time. A set of size 1 = confident; size 2 (`{healthy, bleached}`) = the model is genuinely unsure. This single signal drives the whole MLOps loop: uncertain colonies are flagged in the UI, written to `review_queue`, and the rolling mean set-size is a free **drift detector**. Implemented as hand-rolled LAC (`qhat` = quantile of `1 − p_true` on a held-out calibration split) — transparent and ~15 lines, calibrated on the 1,562-image val split, realized coverage 0.91–0.92 on test.

### 4. The train→inference bridge
Training data is whole-image colony patches; inference classifies SAM2 mask crops. These are reconciled on purpose: **train on the whole patch (resize 224 + ImageNet norm); at inference, crop the mask's bounding box and apply the identical tail.** Because the training distribution (colony-level crops) closely matches what mask bboxes produce, train and inference domains align *by construction* — a tighter match than point-centered crops would give.

### 5. WaterNet over CLAHE
Underwater images suffer wavelength-dependent color loss that simple histogram equalization (CLAHE) can't model. WaterNet is a learned restoration network; it's wired as a pipeline stage (currently identity-passthrough, real weights are vendored as a follow-up) so the enhancement seam exists without blocking the rest.

### 6. 2 classes, not 4 — driven by EDA
The original spec assumed a 4-class CoralNet taxonomy. EDA of the actual NOAA dataset showed only two health states (`CORAL` → healthy, `CORAL_BL` → bleached) — so the model, UI, and conformal sets run **2-class**. `dead` and `algae_covered` stay reserved in the Postgres enum so a future extension needs **no migration**.

### 7. Async inference — forced by a feasibility spike
Before building, a Phase-1.5 spike measured the real pipeline on CPU: **both models resident fit in ~0.9 GB RAM** (so no lazy-loading needed), but SAM2's AMG is the latency bottleneck (~6 s locally → ~45 s on the free Space). A sweep locked the AMG config at **`points_per_side=16` @ 512 px** (the knee of the quality/latency curve). The ~45 s/image reality is *why* `POST /infer` enqueues a job and returns immediately, and the client polls — images and video share one async path.

### 8. Free-tier infrastructure, chosen on constraints
- **Hugging Face Spaces over Render** — Render's free tier (512 MB) can't hold SAM2 + DINOv2; the Space's 16 GB CPU fits comfortably (the spike confirmed ~0.9 GB).
- **Supabase Storage over Cloudflare R2** — reuses the existing Supabase project (one fewer account); the backend still supports R2 by setting four env vars.
- **RLS-locked Supabase** — tables have row-level security enabled with no policies, so the public anon key is denied; only the backend's server-side `service_role` key (which bypasses RLS) can read/write. The frontend never holds a Supabase key.

---

## MLOps

This is a *system*, not just a model.

- **Reproducible training** (`notebooks/01_train_dinov2.ipynb`): one-shot Colab/Kaggle notebook that loads the dataset via its **parquet shards** (4 files vs. 10,419 PNGs — minutes, not hours), checkpoints every epoch, **resumes from the latest checkpoint** across session deaths, logs to W&B, and pushes weights + `conformal.json` to the Hub — all in a single cell so nothing is left to a forgotten manual step.
- **Active-learning flywheel**: conformal set size > 1 → `review_queue` → human confirms in `/admin/review` → `human_labels`. Retraining is **manual** (triggered at 100 new labels) to conserve GPU budget — never auto-wired.
- **Observability from raw SQL** (`/dashboard`): rolling mean prediction-set size (drift proxy), inference latency **p50/p95/p99 + throughput (req/s)**, and class-distribution shift (this week vs. baseline) — all computed from `inference_logs`, no external observability tool.
- **Serving load test** (`backend/loadtest.py`): an asyncio concurrency sweep over the real `/infer` submit→poll→complete path, reporting end-to-end p50/p95/p99 + throughput and exposing tail divergence under load. Measured (Apple Silicon CPU, stub backend): **p99 34 ms → 171 ms** and the widest **p99/p50 gap 2.7× at concurrency 16**, throughput peaking ~226 req/s then saturating. The sweep is surfaced on the dashboard (labeled with its concurrency).
  ```bash
  REEFSCAN_STUB=1 uvicorn backend.main:app --port 8000        # terminal 1
  python -m backend.loadtest --sweep 1,4,8,16,32 --n 64 \      # terminal 2
      --out docs/eval/loadtest.json --machine "your CPU" --stub
  ```
- **Resilient logging**: the Supabase (httpx) client is shared across the event loop and the inference worker thread; on the deployed box this intermittently dropped writes (diagnosed via Postgres logs as client-side, not DB-side). Fixed with a serialize-lock + retry (`persistence._sb_call`) — the kind of production hardening that only shows up once something's actually deployed.
- **Graceful degradation**: with no weights, the API serves contract-valid *stub* output; with no Supabase/Storage, logging/storage no-op. The frontend falls back to mock data when `NEXT_PUBLIC_REEFSCAN_API` is unset — so every layer runs standalone for development.

---

## Serving & inference

An ML-systems layer over the trained model — real serving metrics, a self-hosted VLM baseline, and a
hand-written CUDA kernel. Rationale for the non-obvious calls is in [`DECISIONS.md`](DECISIONS.md).

- **Latency percentiles + throughput under load.** Observability computes **p50/p95/p99 + throughput
  (req/s)** over a window (`backend/observability.py`), and `backend/loadtest.py` drives the real
  `/infer` submit→poll→complete path at rising concurrency. Measured (Apple-Silicon CPU, stub backend):
  **p99 34 → 171 ms**, widest **p99/p50 gap 2.7× at concurrency 16**, throughput peaking ~226 req/s then
  saturating. Surfaced on `/dashboard`, labeled with the concurrency it was measured at.
- **Fused CUDA preprocessing kernel** (`notebooks/serving_B_cuda_kernel.ipynb`, Part B). One kernel fuses
  `uint8 HWC → float32 NCHW → ImageNet-normalize` (4 memory passes → 1). Profiled *first* to state
  preprocessing's latency share, `torch.allclose`-verified against the multi-op torch tail, then
  benchmarked vs it. Measured (**A100-80GB**): **2.14× faster** (0.131 → 0.061 ms/batch-32), **392 GB/s**,
  matches to **7e-7**; preprocessing is **0.8%** of the classify step (DINOv2 fwd 14.1 ms), less e2e.
  Honest: bandwidth-bound, small end-to-end win, real kernel-authoring demo.
- **Self-hosted VLM baseline** (Part C). **Qwen2.5-VL-7B** served with vLLM (OpenAI-compatible) re-runs the
  VLM benchmark on the same NOAA test split — a $0, reproducible open-model column **added next to** the
  GPT-4o one: `DINOv2 specialist vs GPT-4o vs Qwen2.5-VL` on accuracy / macro-F1 / ECE.
- **vLLM serving sweep** (Part D). Concurrency 1→32 → p50/p95/p99 latency + throughput curve for the
  self-hosted 7B VLM.
- **Triton (production path).** [`edge/serving/`](edge/serving/) has the `config.pbtxt` (dynamic batching)
  + `docker-compose.yml`; [`edge/serving/RUNPOD.md`](edge/serving/RUNPOD.md) is the turnkey way to run it on
  a real GPU+Docker box with `perf_analyzer` (~15 min, <$1) — deferred there rather than hacked into Colab,
  which has no Docker daemon.

## Repo layout

```
reefscan/
├── backend/
│   ├── main.py              FastAPI: POST /infer, GET /infer/{id}, /health, review/tracker/observability
│   ├── schemas.py           frozen response contract (mirrors frontend/lib/types.ts)
│   ├── jobs.py              async job store + worker (pipeline in a threadpool)
│   ├── persistence.py       Supabase logging + object storage (resilient, no-op fallbacks)
│   ├── observability.py     inference_logs → drift / latency / class-shift aggregations
│   ├── inference/           enhancer · frame_extractor · segmenter (SAM2) · classifier (DINOv2+conformal) · pipeline
│   ├── data/                CoralPatchDataset (imagefolder) · transforms · EDA · label mapping
│   ├── db/                  schema.sql · seed_demo.sql
│   └── tests/               pytest (async contract + observability)
├── frontend/                Next.js 14 — app/{page,admin/review,tracker,dashboard} · lib/api.ts (swap point)
├── notebooks/01_train_dinov2.ipynb   one-shot Colab/Kaggle trainer + conformal calibration
├── deploy/                  Dockerfile + HF Space README
└── CLAUDE.md                full decision log / build journal
```

---

## Run it locally

**Backend** (Python 3.11+):
```bash
cd backend
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"          # SAM2 (no PyPI release)
# from repo root, with a .env (see .env.example):
REEFSCAN_STUB=1 uvicorn backend.main:app --reload                        # stub mode = no weights needed
pytest backend/tests/                                                    # 10 tests
```

**Frontend** (Node 18+):
```bash
cd frontend && npm install
# point at the live API, or omit to use built-in mock data:
echo 'NEXT_PUBLIC_REEFSCAN_API=https://hrishikabra-reefscan-api.hf.space' > .env.local
npm run dev
```

**Train** — open `notebooks/01_train_dinov2.ipynb` in Colab/Kaggle (GPU), set `WANDB_KEY` + `HF_TOKEN`, Run all.

**Database** — run `backend/db/schema.sql` then `backend/db/seed_demo.sql` in the Supabase SQL editor.

---

## Limitations & future work
- **Latency**: SAM2 AMG on free CPU is ~45 s/image. Path forward: ONNX-export + quantize the SAM2 image encoder (the measured bottleneck — *not* DINOv2, which is ~166 ms/patch).
- **WaterNet** is currently an identity passthrough; vendoring real pretrained weights is the next enhancement.
- **2 classes** until a `dead`/`algae_covered`-bearing source (e.g. ReefNet) is added — the schema already supports it.
- **Trained detection head** for SAM2 prompting (vs. the grid AMG) is documented future work.
- ReefScan is also intended as the **specialist baseline** in a future VLM coral-benchmark paper.

## Acknowledgements
Dataset: **NMFS-OSI / NOAA-PIFSC-ESD Coral Bleaching Dataset**. Models: Meta AI's **DINOv2** and **SAM2**. Built on free tiers from Hugging Face, Supabase, Vercel, Kaggle, and Weights & Biases.
