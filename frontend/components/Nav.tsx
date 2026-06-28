"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Analyze", idx: "01" },
  { href: "/admin/review", label: "Review", idx: "02" },
  { href: "/tracker", label: "Tracker", idx: "03" },
  { href: "/dashboard", label: "Dashboard", idx: "04" },
];

export function Nav() {
  const path = usePathname();
  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));

  return (
    <header className="sticky top-0 z-30">
      <div
        className="mx-auto flex w-full max-w-[1180px] items-center justify-between px-5 py-3.5 md:px-8"
        style={{ background: "rgba(3,16,22,0.72)", backdropFilter: "blur(10px)", borderBottom: "1px solid var(--line)" }}
      >
        <Link href="/" className="group flex items-center gap-3">
          <WaveMark />
          <div className="leading-none">
            <span className="font-display text-[19px] font-semibold tracking-tight text-ink">
              Reef<span className="italic" style={{ color: "var(--cyan)" }}>Scan</span>
            </span>
            <div className="readout mt-1">benthic health · conformal</div>
          </div>
        </Link>

        <nav className="flex items-center gap-1">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="group relative rounded-full px-3.5 py-2 transition-colors"
              style={{ color: active(l.href) ? "var(--ink)" : "var(--ink-dim)" }}
            >
              <span className="readout mr-1.5 opacity-50">{l.idx}</span>
              <span className="font-mono text-[13px] tracking-wide">{l.label}</span>
              {active(l.href) && (
                <span
                  className="absolute inset-x-2 -bottom-[1px] h-[2px] rounded-full"
                  style={{ background: "var(--cyan)", boxShadow: "0 0 10px var(--cyan)" }}
                />
              )}
            </Link>
          ))}
        </nav>

        <div className="hidden items-center gap-2 md:flex">
          <span className="live-dot inline-block h-1.5 w-1.5 rounded-full" style={{ background: "var(--flag)" }} />
          <span className="readout">mock data</span>
        </div>
      </div>
    </header>
  );
}

function WaveMark() {
  return (
    <svg width="30" height="30" viewBox="0 0 30 30" fill="none" aria-hidden>
      <circle cx="15" cy="15" r="14" stroke="var(--line)" />
      <path d="M3 17 Q 8 12, 12 17 T 21 17 T 30 17" stroke="var(--cyan)" strokeWidth="1.4" fill="none" />
      <path d="M3 21 Q 8 16, 12 21 T 21 21 T 30 21" stroke="var(--healthy)" strokeWidth="1.1" fill="none" opacity="0.6" />
      <circle cx="15" cy="15" r="14" stroke="var(--cyan)" strokeOpacity="0.25" />
    </svg>
  );
}
