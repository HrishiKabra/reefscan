"use client";
import { useEffect, useState } from "react";
import { confirmLabel, getReviewQueue, labelsConfirmedThisCycle, RETRAIN_THRESHOLD } from "@/lib/api";
import type { CoralClass, ReviewItem } from "@/lib/types";
import { CLASS_META } from "@/lib/types";
import { ClassTag, ConfidenceBar, SectionLabel, WarnGlyph, classColor } from "@/components/ui";

export default function ReviewPage() {
  const [items, setItems] = useState<ReviewItem[] | null>(null);
  const [confirmed, setConfirmed] = useState(labelsConfirmedThisCycle);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => { getReviewQueue().then(setItems); }, []);

  const handle = async (item: ReviewItem, label: CoralClass | "reject") => {
    if (label !== "reject") {
      await confirmLabel(item.id, label);
      setConfirmed((c) => c + 1);
      setToast(`SEG${item.segment_id} → ${CLASS_META[label].label} written to human_labels`);
    } else {
      setToast(`SEG${item.segment_id} rejected (no label written)`);
    }
    setItems((cur) => (cur ?? []).filter((i) => i.id !== item.id));
    setTimeout(() => setToast(null), 2600);
  };

  const progress = Math.min((confirmed / RETRAIN_THRESHOLD) * 100, 100);

  return (
    <div className="pt-6">
      <div className="rise">
        <span className="readout" style={{ color: "var(--flag)" }}>// active-learning queue</span>
        <h1 className="font-display mt-2 text-[clamp(2rem,5vw,3.2rem)] leading-tight text-ink">
          The model isn’t sure. <span className="italic" style={{ color: "var(--flag)" }}>You decide.</span>
        </h1>
        <p className="mt-3 max-w-xl font-body text-[15px] text-ink-dim">
          Every segment whose conformal set holds more than one class lands here. Confirmed
          labels flow to <span className="font-mono text-ink">human_labels</span> and feed the next manual retrain.
        </p>
      </div>

      {/* retrain progress */}
      <div className="rise panel mt-7 p-5" style={{ animationDelay: "0.08s" }}>
        <div className="mb-2.5 flex items-center justify-between">
          <SectionLabel>retrain trigger</SectionLabel>
          <span className="font-mono text-[12px] text-ink-dim tnum">
            <span className="text-ink">{confirmed}</span> / {RETRAIN_THRESHOLD} labels
          </span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full" style={{ background: "var(--surface-inset)" }}>
          <div className="h-full rounded-full transition-all duration-500"
               style={{ width: `${progress}%`, background: "var(--cyan)", boxShadow: "0 0 12px var(--cyan)" }} />
        </div>
        <p className="mt-2 font-mono text-[11px] text-ink-faint">
          retrain is manual — fires at {RETRAIN_THRESHOLD} new labels to conserve Kaggle GPU budget
        </p>
      </div>

      <div className="mt-7">
        <SectionLabel n="·">pending · {items?.length ?? "…"}</SectionLabel>

        {items && items.length === 0 && (
          <div className="panel flex flex-col items-center px-6 py-16 text-center">
            <div className="font-display text-2xl" style={{ color: "var(--healthy)" }}>Queue clear ✓</div>
            <p className="mt-2 font-mono text-[13px] text-ink-dim">No uncertain segments awaiting review.</p>
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {(items ?? []).map((item, i) => (
            <ReviewCard key={item.id} item={item} index={i} onAction={handle} />
          ))}
          {!items && <SkeletonRows />}
        </div>
      </div>

      {toast && (
        <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2 panel px-4 py-3"
             style={{ borderColor: "var(--cyan)" }}>
          <span className="font-mono text-[12.5px] text-ink">{toast}</span>
        </div>
      )}
    </div>
  );
}

function ReviewCard({
  item, index, onAction,
}: {
  item: ReviewItem;
  index: number;
  onAction: (i: ReviewItem, l: CoralClass | "reject") => void;
}) {
  const top = item.confidence_scores.healthy >= item.confidence_scores.bleached ? "healthy" : "bleached";
  return (
    <div className="rise panel overflow-hidden p-4" style={{ animationDelay: `${0.05 * index + 0.15}s` }}>
      <div className="flex gap-4">
        <PatchPreview seed={item.segment_id} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[12px] text-ink-dim">
              {item.image_id.slice(0, 16)}… · SEG<span className="text-ink">{item.segment_id}</span>
            </span>
            <span className="flag-ring inline-flex items-center gap-1 font-mono text-[11px]" style={{ color: "var(--flag)" }}>
              <WarnGlyph /> set {item.prediction_set.length}
            </span>
          </div>
          <div className="mt-1 font-mono text-[11px] text-ink-faint">{item.reef_location}</div>

          <div className="mt-3">
            <ConfidenceBar scores={item.confidence_scores} />
            <div className="mt-1.5 flex justify-between font-mono text-[11px] tnum">
              <span style={{ color: classColor("healthy") }}>healthy {(item.confidence_scores.healthy * 100).toFixed(0)}%</span>
              <span style={{ color: classColor("bleached") }}>bleached {(item.confidence_scores.bleached * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 border-t pt-3.5 hairline">
        <div className="readout mb-2">confirm true label</div>
        <div className="flex flex-wrap gap-2">
          {(["healthy", "bleached"] as CoralClass[]).map((c) => (
            <button
              key={c}
              onClick={() => onAction(item, c)}
              className="flex-1 rounded-full px-3 py-2 font-mono text-[12px] uppercase tracking-wider transition-transform hover:-translate-y-0.5"
              style={{
                color: classColor(c),
                background: "color-mix(in srgb, " + classColor(c) + " 14%, transparent)",
                border: `1px solid color-mix(in srgb, ${classColor(c)} 45%, transparent)`,
                fontWeight: c === top ? 600 : 400,
              }}
            >
              {CLASS_META[c].label}{c === top ? " ◄" : ""}
            </button>
          ))}
          <button
            onClick={() => onAction(item, "reject")}
            className="rounded-full px-3 py-2 font-mono text-[12px] uppercase tracking-wider text-ink-faint hover:text-ink"
            style={{ border: "1px solid var(--line)" }}
          >
            skip
          </button>
        </div>
      </div>
    </div>
  );
}

// synthesized coral patch (patch_url is a placeholder in mock)
function PatchPreview({ seed }: { seed: number }) {
  const hue = 150 + ((seed * 47) % 60);
  return (
    <div className="relative h-24 w-24 shrink-0 overflow-hidden rounded-lg" style={{ border: "1px solid var(--line)" }}>
      <svg viewBox="0 0 60 60" className="h-full w-full">
        <defs>
          <radialGradient id={`p${seed}`} cx="40%" cy="30%" r="75%">
            <stop offset="0%" stopColor={`hsl(${hue} 55% 42%)`} />
            <stop offset="100%" stopColor="#06222a" />
          </radialGradient>
        </defs>
        <rect width="60" height="60" fill={`url(#p${seed})`} />
        {Array.from({ length: 5 }, (_, i) => (
          <circle key={i} cx={12 + ((seed * (i + 3)) % 38)} cy={14 + ((seed * (i + 7)) % 36)}
                  r={3 + ((seed + i) % 5)} fill="none" stroke="rgba(255,255,255,0.16)" strokeWidth="0.7" />
        ))}
      </svg>
      <span className="absolute bottom-1 left-1 readout" style={{ fontSize: "0.5rem" }}>patch</span>
    </div>
  );
}

function SkeletonRows() {
  return (
    <>
      {[0, 1].map((i) => (
        <div key={i} className="panel scan-sweep h-44 p-4" style={{ opacity: 0.5 }} />
      ))}
    </>
  );
}
