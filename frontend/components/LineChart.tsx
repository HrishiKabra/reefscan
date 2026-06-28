"use client";
import { useState } from "react";
import type { HealthSnapshot } from "@/lib/types";
import { classColor } from "./ui";

const W = 760, H = 320, PL = 44, PR = 18, PT = 22, PB = 34;
const IW = W - PL - PR;
const IH = H - PT - PB;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function monthLabel(iso: string) {
  const m = parseInt(iso.slice(5, 7), 10) - 1;
  return MONTHS[m] + (iso.slice(5, 7) === "01" ? " " + iso.slice(2, 4) : "");
}

export function LineChart({ data }: { data: HealthSnapshot[] }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!data.length) return null;

  const x = (i: number) => PL + (data.length === 1 ? IW / 2 : (i / (data.length - 1)) * IW);
  const y = (v: number) => PT + (1 - v / 100) * IH;

  const line = (key: "healthy_pct" | "bleached_pct") =>
    data.map((d, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(d[key]).toFixed(1)}`).join(" ");

  const areaH =
    `${line("healthy_pct")} L ${x(data.length - 1)} ${y(0)} L ${x(0)} ${y(0)} Z`;

  const hv = hover != null ? data[hover] : null;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id="healthyFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--healthy)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--healthy)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* y gridlines */}
        {[0, 25, 50, 75, 100].map((t) => (
          <g key={t}>
            <line x1={PL} x2={W - PR} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth={1} />
            <text x={PL - 9} y={y(t) + 3} textAnchor="end" fontSize={10} fontFamily="var(--font-mono)" fill="var(--ink-faint)">
              {t}
            </text>
          </g>
        ))}

        {/* x labels */}
        {data.map((d, i) => (
          <text key={i} x={x(i)} y={H - 12} textAnchor="middle" fontSize={9.5}
                fontFamily="var(--font-mono)" fill="var(--ink-faint)">
            {monthLabel(d.date)}
          </text>
        ))}

        {/* healthy area + lines (animated draw) */}
        <path d={areaH} fill="url(#healthyFill)" opacity={0.9} />
        {(["bleached_pct", "healthy_pct"] as const).map((key) => (
          <path
            key={key}
            d={line(key)}
            fill="none"
            stroke={classColor(key === "healthy_pct" ? "healthy" : "bleached")}
            strokeWidth={2.4}
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ strokeDasharray: 1400, strokeDashoffset: 1400, animation: "draw 1.4s ease forwards 0.2s" }}
          />
        ))}

        {/* hover guide */}
        {hv && (
          <line x1={x(hover!)} x2={x(hover!)} y1={PT} y2={PT + IH} stroke="var(--cyan)" strokeWidth={1} strokeDasharray="3 3" />
        )}

        {/* points */}
        {data.map((d, i) =>
          (["healthy_pct", "bleached_pct"] as const).map((key) => (
            <circle
              key={key + i}
              cx={x(i)} cy={y(d[key])} r={hover === i ? 4.5 : 2.6}
              fill="var(--surface)"
              stroke={classColor(key === "healthy_pct" ? "healthy" : "bleached")}
              strokeWidth={2}
              style={{ transition: "r 0.15s" }}
            />
          )),
        )}

        {/* hover hit-zones */}
        {data.map((d, i) => (
          <rect
            key={"hit" + i}
            x={x(i) - IW / (2 * Math.max(data.length - 1, 1))}
            y={PT} width={IW / Math.max(data.length - 1, 1)} height={IH}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
      </svg>

      {/* tooltip */}
      {hv && (
        <div
          className="panel pointer-events-none absolute -translate-x-1/2 p-2.5"
          style={{ left: `${(x(hover!) / W) * 100}%`, top: -6 }}
        >
          <div className="readout mb-1.5">{hv.date}</div>
          <div className="flex gap-3 font-mono text-[12px] tnum">
            <span style={{ color: classColor("healthy") }}>● {hv.healthy_pct.toFixed(1)}%</span>
            <span style={{ color: classColor("bleached") }}>● {hv.bleached_pct.toFixed(1)}%</span>
          </div>
          <div className="mt-1 font-mono text-[11px] text-ink-dim tnum">
            {hv.avg_set_size != null ? `x̄ set ${hv.avg_set_size.toFixed(2)} · ` : ""}n={hv.total_segments}
          </div>
        </div>
      )}
    </div>
  );
}
