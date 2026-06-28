"use client";
import { useState } from "react";

export interface Series { label: string; color: string; values: number[]; dash?: boolean }

const W = 760, H = 240, PL = 48, PR = 16, PT = 16, PB = 30;
const IW = W - PL - PR, IH = H - PT - PB;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function shortDate(iso: string) {
  const m = parseInt(iso.slice(5, 7), 10) - 1;
  return `${MONTHS[m] ?? ""} ${iso.slice(8, 10)}`.trim();
}

/** Generic 1–2 series line chart with hover tooltip. Used for drift + latency. */
export function TimeSeriesChart({
  dates, series, fmt = (n) => `${n}`, yMin, yMax,
}: {
  dates: string[];
  series: Series[];
  fmt?: (n: number) => string;
  yMin?: number;
  yMax?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const all = series.flatMap((s) => s.values);
  const lo = yMin ?? Math.min(...all);
  const hi = yMax ?? Math.max(...all);
  const span = hi - lo || 1;
  const n = dates.length;

  const x = (i: number) => PL + (n <= 1 ? IW / 2 : (i / (n - 1)) * IW);
  const y = (v: number) => PT + (1 - (v - lo) / span) * IH;
  const path = (vals: number[]) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");

  const ticks = [0, 0.5, 1].map((t) => lo + t * span);
  const labelStep = Math.max(1, Math.ceil(n / 7));

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ overflow: "visible" }}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={PL} x2={W - PR} y1={y(t)} y2={y(t)} stroke="var(--line)" />
            <text x={PL - 8} y={y(t) + 3} textAnchor="end" fontSize={10}
                  fontFamily="var(--font-mono)" fill="var(--ink-faint)">{fmt(t)}</text>
          </g>
        ))}
        {dates.map((d, i) => i % labelStep === 0 && (
          <text key={i} x={x(i)} y={H - 10} textAnchor="middle" fontSize={9.5}
                fontFamily="var(--font-mono)" fill="var(--ink-faint)">{shortDate(d)}</text>
        ))}
        {series.map((s) => (
          <path key={s.label} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={2.2}
                strokeLinecap="round" strokeLinejoin="round"
                strokeDasharray={s.dash ? "5 4" : undefined}
                style={{ strokeDasharray: s.dash ? "5 4" : 1600, strokeDashoffset: s.dash ? 0 : 1600,
                         animation: s.dash ? undefined : "draw 1.3s ease forwards 0.15s" }} />
        ))}
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={PT} y2={PT + IH}
                                stroke="var(--cyan)" strokeDasharray="3 3" />}
        {series.map((s) => s.values.map((v, i) => (
          <circle key={s.label + i} cx={x(i)} cy={y(v)} r={hover === i ? 4 : 0}
                  fill="var(--surface)" stroke={s.color} strokeWidth={2} />
        )))}
        {dates.map((_, i) => (
          <rect key={i} x={x(i) - IW / (2 * Math.max(n - 1, 1))} y={PT}
                width={IW / Math.max(n - 1, 1)} height={IH} fill="transparent"
                onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
        ))}
      </svg>
      {hover != null && (
        <div className="panel pointer-events-none absolute -translate-x-1/2 p-2.5"
             style={{ left: `${(x(hover) / W) * 100}%`, top: -4 }}>
          <div className="readout mb-1.5">{dates[hover]}</div>
          {series.map((s) => (
            <div key={s.label} className="flex items-center gap-1.5 font-mono text-[12px] tnum"
                 style={{ color: s.color }}>
              <span className="h-1.5 w-3 rounded-full" style={{ background: s.color }} />
              {s.label} {fmt(s.values[hover])}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
