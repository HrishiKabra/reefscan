# ReefScan

End-to-end underwater image analysis pipeline: segment coral structures, classify health
per segment, quantify uncertainty with conformal prediction, and track reef health over
time. Built to run entirely on free-tier infrastructure.

> **Status:** under construction. This README is a portfolio artifact and will read as a
> technical writeup once the build is complete. Section stubs below mark the decisions
> that get written up — each already has a locked rationale in `CLAUDE.md`.

---

## Architecture (at a glance)

```
upload ──> WaterNet enhance ──> PySceneDetect (video) / passthrough (image)
       ──> SAM2 (Automatic Mask Generator) ──> per-segment centroid patch
       ──> DINOv2-B classifier ──> MAPIE conformal prediction set
       ──> Supabase log + R2 store ──> Next.js dashboard (overlay / review / tracker)
```

- **Backend:** FastAPI on Hugging Face Spaces (free CPU)
- **Frontend:** Next.js 14 + TypeScript + Tailwind on Vercel
- **DB:** Supabase (Postgres) · **Storage:** Cloudflare R2 · **Tracking:** W&B
- **Training:** Kaggle Notebooks (P100, free), weights → Hugging Face Hub

## Health classes
Initial model: `healthy` · `bleached` (from the NOAA-PIFSC bleaching dataset).
`dead` · `algae_covered` are reserved in the DB enum for a future extension — no migration
needed when they arrive.

---

## Design decisions (writeups — TODO, rationale locked in CLAUDE.md)
- [ ] **DINOv2 over a vanilla ViT** — self-supervised features, strong linear-probe transfer
- [ ] **SAM2 over SAM1** — video/temporal masks, better mask quality
- [ ] **Automatic Mask Generator over a trained detection head** — no box labels exist in
      CoralNet; a trained detector is future work
- [ ] **Conformal prediction (MAPIE)** — coverage-guaranteed prediction *sets* drive the
      active-learning queue, not bare point labels
- [ ] **WaterNet over CLAHE** — learned underwater restoration vs. histogram equalization
- [ ] **Why segment AND classify** — segmentation gives spatial display and per-class
      coverage; classification runs on mask-bbox crops. Training on whole colony patches
      (NOAA) closely matches the mask-bbox crops seen at inference, so train/inference
      domains align by construction — a cleaner match than point-centered crops.
- [ ] **HF Spaces over Render** — Render free (512 MB) can't hold the model stack

## MLOps story (TODO)
- [ ] Active-learning flywheel: conformal set size > 1 → `review_queue` → `human_labels` →
      manual retrain at 100 rows
- [ ] Observability: prediction-set-size drift proxy, latency p50/p95, weekly class drift
- [ ] Reproducible training: checkpoint-every-epoch, resume-from-checkpoint, W&B, HF Hub

## Running it
- [ ] Backend setup · [ ] Frontend setup · [ ] Kaggle training notebook · [ ] Demo notes

> ⚠️ **Demo note:** Supabase free tier auto-pauses after ~1 week of inactivity. A cold
> dashboard on first load is expected, not a bug.
