// Atmospheric backdrop: bathymetric contour lines (depth chart) + film grain.
// Pure SVG/CSS, fixed behind all content.
export function Backdrop() {
  // generate concentric-ish contour paths to evoke a depth survey map
  const contours = Array.from({ length: 9 }, (_, i) => {
    const k = i / 8;
    const cx = 78 + k * 6;
    const cy = 26 + k * 5;
    const rx = 14 + i * 8.5;
    const ry = 10 + i * 6.2;
    return { cx, cy, rx, ry, op: 0.10 - k * 0.008 };
  });

  return (
    <>
      <svg
        aria-hidden
        style={{ position: "fixed", inset: 0, width: "100%", height: "100%", zIndex: 0, pointerEvents: "none" }}
        preserveAspectRatio="xMidYMid slice"
        viewBox="0 0 100 100"
      >
        <g fill="none" stroke="var(--cyan)">
          {contours.map((c, i) => (
            <ellipse
              key={i}
              cx={c.cx} cy={c.cy} rx={c.rx} ry={c.ry}
              strokeWidth={0.12}
              style={{ opacity: c.op }}
            />
          ))}
        </g>
        {/* lower-left depth ridges */}
        <g fill="none" stroke="var(--healthy)" strokeWidth={0.1} style={{ opacity: 0.06 }}>
          {Array.from({ length: 7 }, (_, i) => (
            <path
              key={i}
              d={`M -5 ${72 + i * 4.5} Q 25 ${66 + i * 4.5}, 50 ${74 + i * 4.5} T 105 ${70 + i * 4.5}`}
            />
          ))}
        </g>
      </svg>

      {/* film grain */}
      <svg className="grain" aria-hidden>
        <filter id="grain-n">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#grain-n)" />
      </svg>
    </>
  );
}
