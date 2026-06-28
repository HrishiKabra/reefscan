-- ReefScan demo seed — populates a fresh Supabase so the tracker + dashboard render with
-- data before any real uploads. Run AFTER schema.sql, in the Supabase SQL editor.
-- Safe to re-run: it clears the demo rows first (by the fixed reef UUIDs / 'demo%' ids).

delete from inference_logs where image_id like 'demo%';
delete from review_queue where image_id like 'demo%';
delete from health_snapshots where source_image_id like 'demo%';
delete from reef_locations where id in (
  '11111111-1111-1111-1111-111111111111',
  '22222222-2222-2222-2222-222222222222',
  '33333333-3333-3333-3333-333333333333');

-- ---- reef_locations ----
insert into reef_locations (id, name, latitude, longitude, description) values
  ('11111111-1111-1111-1111-111111111111', 'Kāneʻohe Bay — Patch Reef 12', 21.45, -157.79, 'demo'),
  ('22222222-2222-2222-2222-222222222222', 'Molokini Crater', 20.63, -156.49, 'demo'),
  ('33333333-3333-3333-3333-333333333333', 'Hanauma Bay', 21.27, -157.69, 'demo');

-- ---- health_snapshots: 9 monthly surveys per reef, healthy % declining toward now ----
insert into health_snapshots (reef_location_id, request_id, source_image_id, snapshot_time,
  total_segments, healthy_count, bleached_count, dead_count, algae_covered_count,
  total_area_px, healthy_area_px, bleached_area_px, dead_area_px, algae_covered_area_px)
select r.id, gen_random_uuid(), 'demo-survey',
  date_trunc('month', now()) - (m || ' months')::interval,
  20, g.gh, 20 - g.gh, 0, 0,
  200000, (200000 * g.gh / 20.0)::bigint, (200000 * (20 - g.gh) / 20.0)::bigint, 0, 0
from (values
  ('11111111-1111-1111-1111-111111111111'::uuid, 17),
  ('22222222-2222-2222-2222-222222222222'::uuid, 18),
  ('33333333-3333-3333-3333-333333333333'::uuid, 15)
) as r(id, base)
cross join generate_series(0, 8) as m
cross join lateral (
  select greatest(6, least(20, r.base - (8 - m) + (random() * 2 - 1)::int)) as gh
) as g;

-- ---- inference_logs: 14 days x 3 reefs x 6 segments ----
-- Health declines + uncertainty (set size) rises toward now (drift story). Confidences are
-- bimodal (mostly near 0/1) like a real model, so ~10-20% land in the conformal band
-- [1-qhat, qhat] = [0.372, 0.628] and become uncertain. NB the per-row random()s live in a
-- MATERIALIZED CTE — a plain lateral subquery gets hoisted and evaluated ONCE for the whole
-- insert (every row identical). Don't "simplify" it back to a lateral.
insert into inference_logs (request_id, image_id, segment_id, reef_location_id, ts, latency_ms,
  conf_healthy, conf_bleached, conf_dead, conf_algae_covered, prediction_set, prediction_set_size,
  predicted_label, model_version)
with base as materialized (
  select r.id as reef, d, s, random() as u, random() as ra,
    (0.62 - (13 - d) * 0.012) as p_h,   -- healthy prob declines toward now
    (0.08 + (13 - d) * 0.009) as p_u    -- uncertain prob rises toward now (drift)
  from (values
    ('11111111-1111-1111-1111-111111111111'::uuid),
    ('22222222-2222-2222-2222-222222222222'::uuid),
    ('33333333-3333-3333-3333-333333333333'::uuid)
  ) r(id)
  cross join generate_series(0, 13) d
  cross join generate_series(1, 6) s
),
draw as (
  select *,
    case
      when u < p_u       then 0.45 + ra * 0.10   -- uncertain band
      when u < p_u + p_h then 0.70 + ra * 0.28   -- confident healthy
      else                    0.02 + ra * 0.28   -- confident bleached
    end as rh
  from base
)
select gen_random_uuid(), 'demo-' || d || '-' || s, s, reef,
  now() - (d || ' days')::interval - (s || ' minutes')::interval,
  16000 + (13 - d) * 250 + (random() * 4000)::int,
  round(rh::numeric, 3), round((1 - rh)::numeric, 3), 0, 0,
  case when rh between 0.372 and 0.628 then '["healthy","bleached"]'::jsonb
       when rh >= 0.5 then '["healthy"]'::jsonb else '["bleached"]'::jsonb end,
  case when rh between 0.372 and 0.628 then 2 else 1 end,
  (case when rh >= 0.5 then 'healthy' else 'bleached' end)::coral_label,
  'reefscan-dinov2-coral-v1-linearprobe'
from draw;

-- ---- review_queue: a few pending uncertain segments ----
insert into review_queue (request_id, image_id, segment_id, reef_location_id, image_url,
  prediction_set, conf_healthy, conf_bleached, model_version, status)
select gen_random_uuid(), 'demo-review-' || g, g, '11111111-1111-1111-1111-111111111111'::uuid,
  'local://demo.jpg', '["healthy","bleached"]'::jsonb,
  round((0.45 + random() * 0.1)::numeric, 3), round((0.45 + random() * 0.1)::numeric, 3),
  'reefscan-dinov2-coral-v1-linearprobe', 'pending'
from generate_series(1, 5) as g;
