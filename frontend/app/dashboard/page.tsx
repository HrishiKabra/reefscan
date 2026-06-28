"use client";
import { useEffect, useState } from "react";
import { getObservability } from "@/lib/api";
import type { Observability } from "@/lib/types";
import { TimeSeriesChart } from "@/components/TimeSeriesChart";
import { SectionLabel, classColor } from "@/components/ui";

export default function DashboardPage() {
  const [obs, setObs] = useState<Observability | null>(null);
  useEffect(() => { getObservability().then(setObs); }, []);

  const drift = obs?.drift ?? [];
  const latency = obs?.latency ?? [];
  const cd = obs?.class_distribution;
  const driftDates = drift.map((d) => d.date);
  const latDates = latency.map((d) => d.date);

  const driftFirst = drift[0]?.avg_set_size;
  const driftLast = drift[drift.length - 1]?.avg_set_size;
  const driftRising = driftFirst != null && driftLast != null && driftLast - driftFirst > 0.05;
  const lastLat = latency[latency.length - 1];

  return (
    <div className="pt-6">
      <div className="rise">
        <span className="readout" style={{ color: "var(--cyan)" }}>// observability</span>
        <h1 className="font-display mt-2 text-[clamp(2rem,5vw,3.2rem)] leading-tight text-ink">
          Is the model <span className="italic" style={{ color: driftRising ? "var(--flag)" : "var(--healthy)" }}>drifting?</span>
        </h1>
        <p className="mt-3 max-w-xl font-body text-[15px] text-ink-dim">
          Computed entirely from <span className="font-mono text-ink">inference_logs</span> SQL — no
          external tool. Rising mean prediction-set size is the earliest shift signal.
        </p>
      </div>

      {!obs && <div className="panel scan-sweep mt-7 h-64" style={{ opacity: 0.5 }} />}

      {obs && (
        <>
          <div className="rise mt-7 grid grid-cols-2 gap-3 md:grid-cols-4" style={{ animationDelay: "0.06s" }}>
            <Metric label="mean set size (now)" value={driftLast?.toFixed(3) ?? "—"}
                    color={driftRising ? "var(--flag)" : "var(--healthy)"}
                    sub={driftRising ? "▲ rising — drift" : "stable"} />
            <Metric label="latency p50" value={lastLat ? `${(lastLat.p50 / 1000).toFixed(1)}s` : "—"} />
            <Metric label="latency p95" value={lastLat ? `${(lastLat.p95 / 1000).toFixed(1)}s` : "—"} color="var(--bleached)" />
            <Metric label="logs analyzed" value={obs.total_logs.toLocaleString()} />
          </div>

          <div className="rise panel mt-5 p-5 md:p-6" style={{ animationDelay: "0.1s" }}>
            <SectionLabel n="01">prediction-set size · drift proxy</SectionLabel>
            <TimeSeriesChart
              dates={driftDates}
              series={[{ label: "x̄ set", color: "var(--cyan)", values: drift.map((d) => d.avg_set_size) }]}
              yMin={1} yMax={2} fmt={(n) => n.toFixed(2)}
            />
          </div>

          <div className="rise panel mt-5 p-5 md:p-6" style={{ animationDelay: "0.14s" }}>
            <div className="mb-3 flex items-center justify-between">
              <SectionLabel n="02">inference latency p50 / p95</SectionLabel>
              <div className="flex gap-4 font-mono text-[11.5px]">
                <span style={{ color: "var(--cyan)" }}>● p50</span>
                <span style={{ color: "var(--bleached)" }}>● p95</span>
              </div>
            </div>
            <TimeSeriesChart
              dates={latDates}
              series={[
                { label: "p50", color: "var(--cyan)", values: latency.map((d) => d.p50) },
                { label: "p95", color: "var(--bleached)", values: latency.map((d) => d.p95) },
              ]}
              yMin={0}
              fmt={(n) => `${(n / 1000).toFixed(0)}s`}
            />
          </div>

          {cd && (
            <div className="rise panel mt-5 p-5 md:p-6" style={{ animationDelay: "0.18s" }}>
              <SectionLabel n="03">class distribution · this week vs baseline</SectionLabel>
              <ClassShift cd={cd} />
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Metric({ label, value, color, sub }: { label: string; value: string; color?: string; sub?: string }) {
  return (
    <div className="panel p-4">
      <div className="readout mb-2">{label}</div>
      <div className="font-mono text-2xl tnum" style={{ color: color ?? "var(--ink)" }}>{value}</div>
      {sub && <div className="mt-1 font-mono text-[11px]" style={{ color }}>{sub}</div>}
    </div>
  );
}

function ClassShift({ cd }: { cd: NonNullable<Observability["class_distribution"]> }) {
  const classes = Object.keys(cd.current);
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {classes.map((c) => {
        const now = cd.current[c] ?? 0;
        const base = cd.baseline[c] ?? 0;
        const delta = +(now - base).toFixed(1);
        const col = c === "healthy" ? classColor("healthy") : classColor("bleached");
        return (
          <div key={c} className="rounded-lg p-3.5" style={{ border: "1px solid var(--line)" }}>
            <div className="mb-2 flex items-center justify-between">
              <span className="font-mono text-[13px] capitalize" style={{ color: col }}>{c}</span>
              <span className="font-mono text-[12px] tnum"
                    style={{ color: delta === 0 ? "var(--ink-dim)" : delta > 0 ? col : "var(--flag)" }}>
                {delta > 0 ? "+" : ""}{delta}%
              </span>
            </div>
            <Bar label="now" pct={now} color={col} />
            <Bar label="base" pct={base} color="var(--ink-faint)" />
          </div>
        );
      })}
    </div>
  );
}

function Bar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="mb-1.5 flex items-center gap-2">
      <span className="readout w-9">{label}</span>
      <div className="h-2.5 flex-1 overflow-hidden rounded-full" style={{ background: "var(--surface-inset)" }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="w-12 text-right font-mono text-[11px] tnum" style={{ color }}>{pct.toFixed(1)}%</span>
    </div>
  );
}
