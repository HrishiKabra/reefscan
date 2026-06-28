"use client";
import type { InferenceResponse, Segment } from "@/lib/types";
import { classColor } from "./ui";

// Annotated viewer: the source image (or a synthesized reef fallback) with SAM2 segment
// bboxes drawn in the source coordinate space, color-coded by class. Uncertain segments
// (prediction_set_size > 1) get a pulsing dashed alarm ring. Hover/selection sync with
// the segment list via the callbacks.
export function SegmentOverlay({
  data, activeId, onHover,
}: {
  data: InferenceResponse;
  activeId: number | null;
  onHover: (id: number | null) => void;
}) {
  const W = data.image_width ?? 600;
  const H = data.image_height ?? 380;

  return (
    <div className="panel scan-sweep relative overflow-hidden" style={{ aspectRatio: `${W} / ${H}` }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="reefbg" x1="0" y1="0" x2="0.3" y2="1">
            <stop offset="0%" stopColor="#0a3946" />
            <stop offset="55%" stopColor="#082b35" />
            <stop offset="100%" stopColor="#04181f" />
          </linearGradient>
          <radialGradient id="caustic" cx="40%" cy="0%" r="80%">
            <stop offset="0%" stopColor="rgba(98,234,214,0.18)" />
            <stop offset="60%" stopColor="rgba(98,234,214,0)" />
          </radialGradient>
        </defs>

        {/* fallback synthesized reef scene (image_url is a placeholder in mock) */}
        <rect width={W} height={H} fill="url(#reefbg)" />
        <rect width={W} height={H} fill="url(#caustic)" />
        {/* try to load the real image on top; if it 404s the gradient remains */}
        <image href={data.image_url} width={W} height={H} preserveAspectRatio="xMidYMid slice" opacity={0.9} />

        {/* segments */}
        {data.segments.map((s) => (
          <SegmentShape
            key={s.segment_id}
            s={s}
            active={activeId === s.segment_id}
            dim={activeId !== null && activeId !== s.segment_id}
            onHover={onHover}
          />
        ))}
      </svg>

      <div className="pointer-events-none absolute left-3 top-3">
        <span className="readout rounded-md px-2 py-1" style={{ background: "rgba(3,16,22,0.6)" }}>
          ▦ {data.segments.length} segments · SAM2 AMG
        </span>
      </div>
    </div>
  );
}

function SegmentShape({
  s, active, dim, onHover,
}: {
  s: Segment;
  active: boolean;
  dim: boolean;
  onHover: (id: number | null) => void;
}) {
  const [x0, y0, x1, y1] = s.bbox;
  const w = x1 - x0;
  const h = y1 - y0;
  const uncertain = s.prediction_set_size > 1;
  const col = uncertain ? "var(--flag)" : classColor(s.predicted_class);
  const opacity = dim ? 0.32 : 1;

  return (
    <g
      style={{ opacity, transition: "opacity 0.25s", cursor: "pointer" }}
      onMouseEnter={() => onHover(s.segment_id)}
      onMouseLeave={() => onHover(null)}
    >
      <rect
        x={x0} y={y0} width={w} height={h} rx={7}
        fill={col}
        fillOpacity={active ? 0.22 : 0.1}
        stroke={col}
        strokeWidth={active ? 2.4 : 1.6}
        strokeDasharray={uncertain ? "7 5" : undefined}
        className={uncertain ? "flag-ring" : undefined}
      />
      {/* id tab */}
      <g transform={`translate(${x0}, ${y0 - 2})`}>
        <rect x={0} y={-16} width={uncertain ? 40 : 22} height={16} rx={4} fill={col} />
        <text x={5} y={-4} fontSize={10} fontFamily="var(--font-mono)" fill="#03141a" fontWeight={600}>
          {uncertain ? `!${s.segment_id}` : s.segment_id}
        </text>
      </g>
    </g>
  );
}
