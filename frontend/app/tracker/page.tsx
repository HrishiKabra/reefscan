"use client";
import { useEffect, useState } from "react";
import { getReefLocations, getSnapshots } from "@/lib/api";
import type { HealthSnapshot, ReefLocation } from "@/lib/types";
import { LineChart } from "@/components/LineChart";
import { SectionLabel, classColor } from "@/components/ui";

export default function TrackerPage() {
  const [locations, setLocations] = useState<ReefLocation[]>([]);
  const [reefId, setReefId] = useState<string>("");
  const [snaps, setSnaps] = useState<HealthSnapshot[] | null>(null);

  useEffect(() => {
    getReefLocations().then((ls) => {
      setLocations(ls);
      setReefId(ls[0]?.id ?? "");
    });
  }, []);

  useEffect(() => {
    if (!reefId) return;
    setSnaps(null);
    getSnapshots(reefId).then(setSnaps);
  }, [reefId]);

  const reef = locations.find((l) => l.id === reefId);
  const first = snaps?.[0];
  const last = snaps?.[snaps.length - 1];
  const healthDelta = first && last ? +(last.healthy_pct - first.healthy_pct).toFixed(1) : 0;
  const hasDrift = first?.avg_set_size != null && last?.avg_set_size != null;
  const driftDelta = hasDrift ? +(last!.avg_set_size! - first!.avg_set_size!).toFixed(2) : 0;

  return (
    <div className="pt-6">
      <div className="rise">
        <span className="readout" style={{ color: "var(--cyan)" }}>// temporal tracker</span>
        <h1 className="font-display mt-2 text-[clamp(2rem,5vw,3.2rem)] leading-tight text-ink">
          Health, <span className="italic" style={{ color: "var(--healthy)" }}>over time.</span>
        </h1>
        <p className="mt-3 max-w-xl font-body text-[15px] text-ink-dim">
          Area-weighted class coverage per reef across repeated surveys. A rising mean
          prediction-set size is an early drift signal.
        </p>
      </div>

      {/* location selector */}
      <div className="rise mt-7 flex flex-wrap gap-2" style={{ animationDelay: "0.06s" }}>
        {locations.map((l) => {
          const on = l.id === reefId;
          return (
            <button
              key={l.id}
              onClick={() => setReefId(l.id)}
              className="rounded-full px-4 py-2 font-mono text-[12.5px] transition-colors"
              style={{
                color: on ? "#fff" : "var(--ink-dim)",
                background: on ? "var(--cyan)" : "transparent",
                border: `1px solid ${on ? "var(--cyan)" : "var(--line)"}`,
                fontWeight: on ? 600 : 400,
              }}
            >
              {l.name}
            </button>
          );
        })}
      </div>

      {/* stat row */}
      <div className="rise mt-5 grid grid-cols-2 gap-3 md:grid-cols-4" style={{ animationDelay: "0.1s" }}>
        <Metric label="current healthy" value={last ? `${last.healthy_pct.toFixed(1)}%` : "—"} color={classColor("healthy")} />
        <Metric
          label="Δ since baseline"
          value={`${healthDelta > 0 ? "+" : ""}${healthDelta}%`}
          color={healthDelta < 0 ? "var(--bleached)" : "var(--healthy)"}
          sub={healthDelta < 0 ? "▼ declining" : "▲ improving"}
        />
        <Metric label="x̄ set size" value={hasDrift ? last!.avg_set_size!.toFixed(2) : "—"}
                color={hasDrift && last!.avg_set_size! > 1.2 ? "var(--flag)" : "var(--ink)"}
                sub={hasDrift ? undefined : "see dashboard"} />
        <Metric
          label="drift (Δ set)"
          value={hasDrift ? `${driftDelta > 0 ? "+" : ""}${driftDelta}` : "—"}
          color={hasDrift && driftDelta > 0.1 ? "var(--flag)" : "var(--ink-dim)"}
          sub={hasDrift ? (driftDelta > 0.1 ? "⚠ shift" : "stable") : "→ /dashboard"}
        />
      </div>

      {/* chart */}
      <div className="rise panel mt-5 p-5 md:p-6" style={{ animationDelay: "0.14s" }}>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <SectionLabel>{reef ? reef.name : "—"} · area-weighted %</SectionLabel>
          <Legend />
        </div>
        {snaps ? <LineChart data={snaps} /> : <div className="scan-sweep h-[260px] rounded-lg" style={{ background: "var(--surface-inset)" }} />}
        {reef && (
          <div className="mt-3 font-mono text-[11px] text-ink-faint tnum">
            {reef.lat.toFixed(2)}°, {reef.lng.toFixed(2)}° · {snaps?.length ?? 0} surveys
          </div>
        )}
      </div>
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

function Legend() {
  return (
    <div className="flex items-center gap-4 font-mono text-[11.5px]">
      <span className="flex items-center gap-1.5" style={{ color: classColor("healthy") }}>
        <span className="h-2 w-4 rounded-full" style={{ background: classColor("healthy") }} /> healthy
      </span>
      <span className="flex items-center gap-1.5" style={{ color: classColor("bleached") }}>
        <span className="h-2 w-4 rounded-full" style={{ background: classColor("bleached") }} /> bleached
      </span>
    </div>
  );
}
