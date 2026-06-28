# ReefScan — UI/UX design brief

Hand this to a design pass (e.g. Claude) **with the repo linked as context**. Goal: take a
functional, already-themed frontend and elevate it to a polished, portfolio-grade, fully
responsive interface — *without changing any data contracts or breaking the live backend
integration.*

## What ReefScan is
A coral-reef health analysis tool. Users upload an underwater photo; the backend segments
coral colonies, classifies each **healthy / bleached**, and returns a **conformal prediction
set** per colony (set size 1 = confident, 2 = uncertain). Uncertain colonies feed a human
review queue; all inferences feed observability dashboards. It's a **portfolio project** for
ML/MLOps engineering interviews, so the UI should read as *credible, technical, and
distinctive* — not a generic SaaS template.

## Current state
- **Stack:** Next.js 14 (App Router), TypeScript, Tailwind CSS. No component library (hand-built SVG charts/overlays). Fonts via `next/font`: Fraunces (display), IBM Plex Mono (data/labels), Hanken Grotesk (body).
- **Aesthetic:** "abyssal research instrument" — deep-ocean dark theme, bathymetric contour + grain background, bioluminescent cyan accent, with a health palette that *is* the legend: `--healthy` teal-green, `--bleached` warning amber, `--flag` alarm pink (uncertain). Design tokens live in `frontend/app/globals.css` (CSS variables) and `frontend/tailwind.config.ts`.
- **Four pages** (`frontend/app/`):
  1. `page.tsx` — **Analyze**: drag-drop / file / URL upload, async "pipeline running" loader, annotated SVG segment overlay (`components/SegmentOverlay.tsx`) synced to a segment-card grid (`components/SegmentCard.tsx`), area-weighted health summary.
  2. `admin/review/page.tsx` — **Review queue**: uncertain segments, confidence bars, confirm-label buttons, retrain-progress meter.
  3. `tracker/page.tsx` — **Tracker**: reef-location selector + hand-built line chart (`components/LineChart.tsx`) of health % over time.
  4. `dashboard/page.tsx` — **Observability**: drift / latency / class-distribution charts (`components/TimeSeriesChart.tsx`).

Screenshots of all four are in `docs/screenshots/`.

## Keep (don't throw away the identity)
- The dark "instrument" direction, the serif+mono pairing, and the health-as-legend color system. These are intentional and differentiate it. Refine, don't replace.
- The hand-built SVG charts/overlay (no heavy chart dep).

## Improve (the asks)
1. **Responsiveness / mobile** — the layouts are desktop-first; make all four pages genuinely good on phone + tablet (the segment overlay + card grid, the charts, the nav).
2. **Polish & hierarchy** — tighten spacing, typographic rhythm, and visual hierarchy; make the Analyze results page feel like a considered report, not a dump of cards.
3. **States** — design proper **loading, empty, and error** states everywhere (e.g. API down, job failed, empty review queue, Supabase cold-start). Today some are minimal.
4. **The overlay viewer** is the hero — make segment hover/selection, the uncertain-flag treatment, and the confidence display as crisp and legible as possible.
5. **Micro-interactions & motion** — purposeful, performant (prefer CSS / the existing keyframes); one well-orchestrated page-load reveal beats scattered effects.
6. **Accessibility** — color contrast (especially amber/pink on dark), focus states, keyboard nav, `prefers-reduced-motion`, semantic landmarks, alt text.
7. **A real landing/hero** for the Analyze page that explains the product in one glance for first-time visitors (recruiters).

## Hard constraints — do NOT break
- **Do not change the data contracts.** `frontend/lib/types.ts` mirrors the backend response shape exactly; `frontend/lib/api.ts` is the single integration point (real backend when `NEXT_PUBLIC_REEFSCAN_API` is set, else mock). Reshape UI, not the API calls or types.
- Keep it **Next.js 14 App Router + Tailwind**. A lightweight, well-justified addition (e.g. `framer-motion`, `clsx`) is fine; avoid a heavy component library that fights the bespoke aesthetic.
- Preserve the **4 routes** and their behaviors (async submit+poll on Analyze; confirm→`human_labels` on Review; reef filter on Tracker; the three observability views on Dashboard).
- It must still **build clean** (`npm run build`) and render against the built-in mock data with no backend.
- Backend/ML are out of scope — frontend only.

## Definition of done
All four pages: responsive (mobile→desktop), accessible, with polished loading/empty/error states, building clean, still wired to `lib/api.ts`, and keeping the distinctive abyssal-instrument identity. Bonus: a short note on what changed and why.
