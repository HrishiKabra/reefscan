import type { CoralClass } from "@/lib/types";
import { CLASS_META } from "@/lib/types";

export function classColor(c: CoralClass) {
  return CLASS_META[c].varName;
}

export function ClassTag({ c, dim }: { c: CoralClass; dim?: boolean }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-[11px] uppercase tracking-wider"
      style={{
        color: classColor(c),
        background: dim ? "transparent" : "color-mix(in srgb, " + classColor(c) + " 12%, transparent)",
        border: `1px solid color-mix(in srgb, ${classColor(c)} 38%, transparent)`,
      }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: classColor(c) }} />
      {CLASS_META[c].label}
    </span>
  );
}

export function UncertainTag({ setSize }: { setSize: number }) {
  return (
    <span
      className="flag-ring inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-[11px] uppercase tracking-wider"
      style={{
        color: "var(--flag)",
        background: "color-mix(in srgb, var(--flag) 12%, transparent)",
        border: "1px solid color-mix(in srgb, var(--flag) 45%, transparent)",
      }}
    >
      <WarnGlyph /> uncertain · set {setSize}
    </span>
  );
}

export function WarnGlyph() {
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" fill="none" aria-hidden>
      <path d="M6 1 L11 10.5 H1 Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
      <path d="M6 4.6 V7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      <circle cx="6" cy="8.7" r="0.7" fill="currentColor" />
    </svg>
  );
}

/** Two-class confidence bar (raw softmax). */
export function ConfidenceBar({
  scores, height = 6,
}: {
  scores: Record<CoralClass, number>;
  height?: number;
}) {
  const h = scores.healthy * 100;
  const b = scores.bleached * 100;
  return (
    <div className="w-full overflow-hidden rounded-full" style={{ height, background: "rgba(255,255,255,0.05)" }}>
      <div className="flex h-full w-full">
        <div style={{ width: `${h}%`, background: classColor("healthy") }} />
        <div style={{ width: `${b}%`, background: classColor("bleached") }} />
      </div>
    </div>
  );
}

export function SectionLabel({ children, n }: { children: React.ReactNode; n?: string }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      {n && <span className="readout" style={{ color: "var(--cyan)" }}>{n}</span>}
      <span className="readout">{children}</span>
      <span className="h-px flex-1" style={{ background: "var(--line)" }} />
    </div>
  );
}

export function pct(n: number) {
  return `${n.toFixed(n % 1 === 0 ? 0 : 1)}%`;
}
