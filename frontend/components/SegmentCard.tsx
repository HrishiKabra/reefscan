"use client";
import type { Segment } from "@/lib/types";
import { CLASS_META } from "@/lib/types";
import { ClassTag, ConfidenceBar, UncertainTag, classColor, pct } from "./ui";

export function SegmentCard({
  s, active, onHover, index,
}: {
  s: Segment;
  active: boolean;
  onHover: (id: number | null) => void;
  index: number;
}) {
  const uncertain = s.prediction_set_size > 1;
  return (
    <div
      onMouseEnter={() => onHover(s.segment_id)}
      onMouseLeave={() => onHover(null)}
      className="rise panel cursor-pointer p-3.5 transition-all"
      style={{
        animationDelay: `${0.05 * index + 0.2}s`,
        borderColor: active ? (uncertain ? "var(--flag)" : classColor(s.predicted_class)) : "var(--line)",
        transform: active ? "translateX(3px)" : "none",
        boxShadow: active ? `0 0 0 1px ${uncertain ? "var(--flag)" : classColor(s.predicted_class)}` : "none",
      }}
    >
      <div className="flex items-center justify-between">
        <span className="font-mono text-[12px] text-ink-dim">
          SEG<span className="text-ink">{String(s.segment_id).padStart(2, "0")}</span>
        </span>
        {uncertain ? <UncertainTag setSize={s.prediction_set_size} /> : <ClassTag c={s.predicted_class} />}
      </div>

      <div className="mt-3">
        <ConfidenceBar scores={s.confidence_scores} />
        <div className="mt-1.5 flex justify-between font-mono text-[11px] tnum">
          <span style={{ color: classColor("healthy") }}>H {(s.confidence_scores.healthy * 100).toFixed(0)}</span>
          <span style={{ color: classColor("bleached") }}>B {(s.confidence_scores.bleached * 100).toFixed(0)}</span>
        </div>
      </div>

      <div className="mt-2.5 flex items-center justify-between border-t pt-2.5 hairline">
        <span className="font-mono text-[11px] text-ink-faint">
          set [{s.prediction_set.map((c) => CLASS_META[c].short).join(" · ")}]
        </span>
        <span className="font-mono text-[11px] text-ink-dim tnum">cover {pct(s.coverage_pct)}</span>
      </div>
    </div>
  );
}
