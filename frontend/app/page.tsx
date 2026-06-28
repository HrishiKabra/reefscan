"use client";
import { useCallback, useRef, useState } from "react";
import { getInferenceResult, submitJob } from "@/lib/api";
import type { InferenceResponse } from "@/lib/types";
import { SegmentOverlay } from "@/components/SegmentOverlay";
import { SegmentCard } from "@/components/SegmentCard";
import { ClassTag, SectionLabel, classColor, pct } from "@/components/ui";

type View = "idle" | "loading" | "done";
const STAGES = ["WaterNet enhance", "Scene frames", "SAM2 segment", "DINOv2 classify", "Conformal sets"];

export default function AnalyzePage() {
  const [view, setView] = useState<View>("idle");
  const [data, setData] = useState<InferenceResponse | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const [drag, setDrag] = useState(false);
  const [url, setUrl] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const run = useCallback(async (input: File | string) => {
    setView("loading");
    const { job_id } = await submitJob(input);
    const res = await getInferenceResult(job_id);
    setData(res);
    setView("done");
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) run(f);
  };

  return (
    <div>
      <Header />

      {view === "idle" && (
        <section
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          className="rise panel relative mt-8 overflow-hidden"
          style={{ borderStyle: "dashed", borderColor: drag ? "var(--cyan)" : "var(--line)" }}
        >
          <div className="flex flex-col items-center px-6 py-16 text-center">
            <DropGlyph />
            <h2 className="font-display mt-5 text-2xl text-ink">Drop a reef image or clip</h2>
            <p className="mt-2 max-w-md font-body text-sm text-ink-dim">
              Photos or short transects. The pipeline runs async — enhance, segment, classify,
              and wrap each colony in a conformal prediction set.
            </p>

            <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
              <button
                onClick={() => fileRef.current?.click()}
                className="rounded-full px-5 py-2.5 font-mono text-[13px] uppercase tracking-wider transition-transform hover:-translate-y-0.5"
                style={{ background: "var(--cyan)", color: "#03141a", fontWeight: 600 }}
              >
                Choose file
              </button>
              <button
                onClick={() => run("__sample__")}
                className="rounded-full border px-5 py-2.5 font-mono text-[13px] uppercase tracking-wider text-ink transition-colors hover:text-cyan"
                style={{ borderColor: "var(--line)" }}
              >
                Run sample ▶
              </button>
              <input ref={fileRef} type="file" accept="image/*,video/*" hidden
                     onChange={(e) => e.target.files?.[0] && run(e.target.files[0])} />
            </div>

            <div className="mt-6 flex w-full max-w-md items-center gap-2">
              <span className="h-px flex-1" style={{ background: "var(--line)" }} />
              <span className="readout">or paste a url</span>
              <span className="h-px flex-1" style={{ background: "var(--line)" }} />
            </div>
            <form
              onSubmit={(e) => { e.preventDefault(); if (url.trim()) run(url.trim()); }}
              className="mt-3 flex w-full max-w-md items-center gap-2"
            >
              <input
                value={url} onChange={(e) => setUrl(e.target.value)}
                placeholder="https://…/reef.jpg"
                className="flex-1 rounded-full bg-transparent px-4 py-2.5 font-mono text-[13px] text-ink outline-none"
                style={{ border: "1px solid var(--line)" }}
              />
              <button className="rounded-full px-4 py-2.5 font-mono text-[13px] text-ink-dim hover:text-cyan" type="submit">
                go →
              </button>
            </form>
          </div>
        </section>
      )}

      {view === "loading" && <LoadingConsole />}

      {view === "done" && data && (
        <Results data={data} active={active} setActive={setActive} reset={() => { setView("idle"); setData(null); }} />
      )}
    </div>
  );
}

function Header() {
  return (
    <div className="rise pt-6">
      <span className="readout" style={{ color: "var(--cyan)" }}>// benthic survey console</span>
      <h1 className="font-display mt-2 text-[clamp(2.4rem,6vw,4rem)] leading-[0.98] text-ink">
        Read the reef,<br />
        <span className="italic" style={{ color: "var(--healthy)" }}>quantify</span> the doubt.
      </h1>
      <p className="mt-4 max-w-xl font-body text-[15px] leading-relaxed text-ink-dim">
        Upload an underwater image. ReefScan segments coral colonies, classifies each as
        healthy or bleached, and returns a <span className="text-ink">conformal prediction set</span> —
        so uncertainty is visible, not hidden behind a single label.
      </p>
    </div>
  );
}

function Results({
  data, active, setActive, reset,
}: {
  data: InferenceResponse;
  active: number | null;
  setActive: (n: number | null) => void;
  reset: () => void;
}) {
  const s = data.summary;
  return (
    <div className="mt-8">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <SectionLabel n="01">analysis complete</SectionLabel>
          <div className="flex items-center gap-3 font-mono text-[12px] text-ink-dim">
            <span>job {data.job_id.slice(0, 8)}</span>
            <span style={{ color: "var(--line)" }}>|</span>
            <span>{(data.processing_time_ms / 1000).toFixed(1)}s</span>
            <span style={{ color: "var(--line)" }}>|</span>
            <span style={{ color: "var(--cyan)" }}>{data.model_version}</span>
          </div>
        </div>
        <button onClick={reset} className="rounded-full border px-4 py-2 font-mono text-[12px] uppercase tracking-wider text-ink-dim hover:text-cyan"
                style={{ borderColor: "var(--line)" }}>
          ↺ new scan
        </button>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.55fr_1fr]">
        <div className="rise">
          <SegmentOverlay data={data} activeId={active} onHover={setActive} />
          {s.uncertain_segments > 0 && (
            <div className="mt-3 flex items-center gap-2 rounded-lg px-3.5 py-2.5"
                 style={{ background: "color-mix(in srgb, var(--flag) 9%, transparent)", border: "1px solid color-mix(in srgb, var(--flag) 35%, transparent)" }}>
              <span className="flag-ring h-2 w-2 rounded-full" style={{ background: "var(--flag)" }} />
              <span className="font-mono text-[12.5px]" style={{ color: "var(--flag)" }}>
                {s.uncertain_segments} uncertain segment{s.uncertain_segments > 1 ? "s" : ""} auto-flagged to the review queue (set size &gt; 1)
              </span>
            </div>
          )}
        </div>

        <div className="rise" style={{ animationDelay: "0.1s" }}>
          <SummaryPanel data={data} />
        </div>
      </div>

      <div className="mt-7">
        <SectionLabel n="02">segments · {data.segments.length}</SectionLabel>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {data.segments.map((seg, i) => (
            <SegmentCard key={seg.segment_id} s={seg} index={i} active={active === seg.segment_id} onHover={setActive} />
          ))}
        </div>
      </div>
    </div>
  );
}

function SummaryPanel({ data }: { data: InferenceResponse }) {
  const s = data.summary;
  const hp = s.area_weighted.healthy_pct;
  return (
    <div className="panel p-5">
      <SectionLabel>area-weighted health</SectionLabel>

      <div className="flex items-end gap-2">
        <span className="font-display text-5xl leading-none tnum" style={{ color: classColor("healthy") }}>
          {hp.toFixed(0)}
        </span>
        <span className="mb-1 font-mono text-sm text-ink-dim">% healthy cover</span>
      </div>

      <div className="mt-4 flex h-3 w-full overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.05)" }}>
        <div style={{ width: `${hp}%`, background: classColor("healthy") }} />
        <div style={{ width: `${s.area_weighted.bleached_pct}%`, background: classColor("bleached") }} />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[11px] tnum">
        <span style={{ color: classColor("healthy") }}>healthy {pct(hp)}</span>
        <span style={{ color: classColor("bleached") }}>bleached {pct(s.area_weighted.bleached_pct)}</span>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3">
        <Stat label="total segments" value={s.total_segments} />
        <Stat label="uncertain" value={s.uncertain_segments} accent="var(--flag)" />
        <Stat label="dominant" valueNode={<ClassTag c={s.dominant_status} />} />
        <Stat label="latency" value={`${(data.processing_time_ms / 1000).toFixed(1)}s`} />
      </div>
    </div>
  );
}

function Stat({
  label, value, valueNode, accent,
}: {
  label: string; value?: string | number; valueNode?: React.ReactNode; accent?: string;
}) {
  return (
    <div className="rounded-lg p-3" style={{ background: "rgba(255,255,255,0.025)", border: "1px solid var(--line)" }}>
      <div className="readout mb-1.5">{label}</div>
      {valueNode ?? (
        <div className="font-mono text-xl tnum" style={{ color: accent ?? "var(--ink)" }}>{value}</div>
      )}
    </div>
  );
}

function LoadingConsole() {
  return (
    <section className="rise panel mt-8 p-8">
      <SectionLabel n="··">pipeline running</SectionLabel>
      <div className="mx-auto max-w-md">
        {STAGES.map((st, i) => (
          <div key={st} className="flex items-center gap-3 py-2.5">
            <span className="font-mono text-[11px] text-ink-faint">{String(i + 1).padStart(2, "0")}</span>
            <span className="relative h-1.5 flex-1 overflow-hidden rounded-full scan-sweep"
                  style={{ background: "rgba(255,255,255,0.05)", animationDelay: `${i * 0.2}s` }} />
            <span className="font-mono text-[12.5px] text-ink-dim">{st}</span>
          </div>
        ))}
        <p className="mt-4 text-center font-mono text-[11px] text-ink-faint">
          async job · ~15–25s on free CPU
        </p>
      </div>
    </section>
  );
}

function DropGlyph() {
  return (
    <svg width="46" height="46" viewBox="0 0 46 46" fill="none" aria-hidden>
      <rect x="1" y="1" width="44" height="44" rx="12" stroke="var(--line)" />
      <path d="M23 13 V29 M16 22 L23 29 L30 22" stroke="var(--cyan)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M13 33 H33" stroke="var(--healthy)" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
