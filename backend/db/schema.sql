-- ReefScan — Supabase (Postgres) schema
-- Phase 1 deliverable.
--
-- Tables:
--   jobs             — async inference jobs (POST /infer enqueues, GET /infer/{job_id} polls)
--   reef_locations   — named reef sites; uploads attach to one
--   inference_logs   — one row per classified SEGMENT (grouped by request_id);
--                      powers observability + drift views
--   review_queue     — uncertain predictions (conformal set size > 1) awaiting human label
--   human_labels     — confirmed labels from /admin/review; feeds manual retraining
--   health_snapshots — per-upload aggregate per reef; powers the temporal tracker
--                      (stores BOTH per-class counts and per-class pixel areas)
--
-- Conventions: uuid PKs, timestamptz, jsonb for variable-shape fields.
-- Apply via Supabase SQL editor or `supabase db push`.

-- ---------------------------------------------------------------------------
-- Enums
-- ---------------------------------------------------------------------------
-- coral_label keeps all 4 values, but the INITIAL model is 2-class (healthy/bleached);
-- 'dead' and 'algae_covered' are RESERVED for a future extension (ReefNet supplementation)
-- so no enum migration is needed when they arrive. See CLAUDE.md / label_mapping.py.
do $$ begin
  create type coral_label as enum ('healthy', 'bleached', 'dead', 'algae_covered');
exception when duplicate_object then null; end $$;

do $$ begin
  create type review_status as enum ('pending', 'confirmed', 'rejected');
exception when duplicate_object then null; end $$;

do $$ begin
  create type job_status as enum ('queued', 'processing', 'complete', 'failed');
exception when duplicate_object then null; end $$;

-- ---------------------------------------------------------------------------
-- reef_locations  (defined first — referenced by jobs/inference_logs/etc.)
-- ---------------------------------------------------------------------------
create table if not exists reef_locations (
  id           uuid primary key default gen_random_uuid(),
  name         text not null,
  latitude     double precision,
  longitude    double precision,
  description  text,
  created_at   timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- jobs  (async inference; POST /infer enqueues, GET /infer/{job_id} polls)
-- Same pipeline for image and video (a video is just N frames).
-- ---------------------------------------------------------------------------
create table if not exists jobs (
  job_id            uuid primary key default gen_random_uuid(),
  status            job_status not null default 'queued',
  reef_location_id  uuid references reef_locations(id) on delete set null,
  source_kind       text,                              -- 'image' | 'video'
  source_url        text,                              -- R2 url of the uploaded file
  created_at        timestamptz not null default now(),
  completed_at      timestamptz,
  result_json       jsonb,                             -- structured pipeline output when complete
  error_message     text                               -- populated when status = 'failed'
);

create index if not exists idx_jobs_status  on jobs (status);
create index if not exists idx_jobs_created on jobs (created_at);

-- ---------------------------------------------------------------------------
-- inference_logs  (one row per classified segment)
-- ---------------------------------------------------------------------------
create table if not exists inference_logs (
  id                  uuid primary key default gen_random_uuid(),
  request_id          uuid not null,                 -- groups all segments of one /infer call
  image_id            text not null,                 -- R2 object key / source image id
  segment_id          int  not null,                 -- index of the SAM2 mask within the image
  reef_location_id    uuid references reef_locations(id) on delete set null,

  ts                  timestamptz not null default now(),
  latency_ms          integer not null,              -- full-image pipeline latency (same per request)

  -- raw softmax per class (kept as explicit columns for easy SQL aggregation)
  conf_healthy        real not null,
  conf_bleached       real not null,
  conf_dead           real not null,
  conf_algae_covered  real not null,

  -- conformal output
  prediction_set      jsonb not null,                -- e.g. ["healthy","algae_covered"]
  prediction_set_size int  not null,                 -- 1 = confident, >1 = uncertain
  predicted_label     coral_label,                   -- argmax / point label for convenience

  model_version       text not null,

  created_at          timestamptz not null default now()
);

create index if not exists idx_inflogs_ts        on inference_logs (ts);
create index if not exists idx_inflogs_request   on inference_logs (request_id);
create index if not exists idx_inflogs_location  on inference_logs (reef_location_id);
create index if not exists idx_inflogs_setsize   on inference_logs (prediction_set_size);
create index if not exists idx_inflogs_version   on inference_logs (model_version);

-- ---------------------------------------------------------------------------
-- review_queue  (uncertain predictions -> human review)
-- ---------------------------------------------------------------------------
create table if not exists review_queue (
  id                  uuid primary key default gen_random_uuid(),
  request_id          uuid not null,
  image_id            text not null,
  segment_id          int  not null,
  reef_location_id    uuid references reef_locations(id) on delete set null,

  patch_url           text,                          -- R2 url of the centroid patch crop
  image_url           text,                          -- R2 url of the full source image

  prediction_set      jsonb not null,                -- candidate labels shown to the human
  conf_healthy        real,
  conf_bleached       real,
  conf_dead           real,
  conf_algae_covered  real,

  model_version       text not null,
  status              review_status not null default 'pending',

  created_at          timestamptz not null default now()
);

create index if not exists idx_review_status   on review_queue (status);
create index if not exists idx_review_created  on review_queue (created_at);

-- ---------------------------------------------------------------------------
-- human_labels  (confirmed labels; retrain trigger at 100 new rows)
-- ---------------------------------------------------------------------------
create table if not exists human_labels (
  id               uuid primary key default gen_random_uuid(),
  review_queue_id  uuid references review_queue(id) on delete set null,
  image_id         text not null,
  segment_id       int  not null,
  confirmed_label  coral_label not null,
  labeled_by       text,                             -- reviewer id/email
  used_in_training boolean not null default false,   -- flips true once consumed by a retrain
  created_at       timestamptz not null default now()
);

create index if not exists idx_humanlabels_unused  on human_labels (used_in_training);
create index if not exists idx_humanlabels_created on human_labels (created_at);

-- ---------------------------------------------------------------------------
-- health_snapshots  (per-upload aggregate; temporal tracker source)
-- Stores BOTH counts and pixel areas per class so the UI can show count-% or area-%.
-- ---------------------------------------------------------------------------
create table if not exists health_snapshots (
  id                  uuid primary key default gen_random_uuid(),
  reef_location_id    uuid references reef_locations(id) on delete cascade,
  request_id          uuid not null,
  source_image_id     text not null,
  snapshot_time       timestamptz not null default now(),

  total_segments      int  not null,
  healthy_count       int  not null default 0,
  bleached_count      int  not null default 0,
  dead_count          int  not null default 0,
  algae_covered_count int  not null default 0,

  total_area_px            bigint not null default 0,
  healthy_area_px          bigint not null default 0,
  bleached_area_px         bigint not null default 0,
  dead_area_px             bigint not null default 0,
  algae_covered_area_px    bigint not null default 0,

  created_at          timestamptz not null default now()
);

create index if not exists idx_snapshots_location_time
  on health_snapshots (reef_location_id, snapshot_time);
