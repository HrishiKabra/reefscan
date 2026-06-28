"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";

const LINKS = [
  { href: "/", label: "Analyze", num: "01" },
  { href: "/admin/review", label: "Review", num: "02" },
  { href: "/tracker", label: "Tracker", num: "03" },
  { href: "/dashboard", label: "Dashboard", num: "04" },
];

export function Nav() {
  const path = usePathname();
  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));

  return (
    <header className="mx-auto flex w-full max-w-[1240px] flex-wrap items-center justify-between gap-4 px-4 py-5 md:px-8">
      <Link href="/" className="flex items-center gap-3" style={{ textDecoration: "none", color: "inherit" }}>
        <span className="grid h-[38px] w-[38px] place-items-center rounded-xl"
              style={{ background: "var(--surface)", boxShadow: "var(--shadow)", border: "1px solid var(--line)" }}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 13c2-2.4 4-2.4 6 0s4 2.4 6 0 4-2.4 6 0" />
            <path d="M2 18c2-2.4 4-2.4 6 0s4 2.4 6 0 4-2.4 6 0" />
            <circle cx="12" cy="6.5" r="2.4" />
          </svg>
        </span>
        <span className="flex flex-col leading-none">
          <span className="font-display text-[21px] font-medium tracking-tight">
            Reef<span className="italic" style={{ color: "var(--accent)" }}>Scan</span>
          </span>
          <span className="readout mt-1" style={{ fontSize: "9.5px", letterSpacing: "0.22em" }}>Benthic Health</span>
        </span>
      </Link>

      <nav aria-label="Primary" className="flex items-center gap-1 rounded-full p-[5px]"
           style={{ background: "var(--surface)", border: "1px solid var(--line)", boxShadow: "var(--shadow)" }}>
        {LINKS.map((l) => {
          const on = active(l.href);
          return (
            <Link key={l.href} href={l.href}
              className="flex items-center gap-1.5 rounded-full px-3.5 py-2 text-[13px] font-semibold transition-colors"
              style={{
                color: on ? "#fff" : "var(--ink-2s)",
                background: on ? "var(--accent)" : "transparent",
              }}>
              <span className="font-mono text-[10px] opacity-60">{l.num}</span>
              <span>{l.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="flex items-center gap-2.5">
        <span className="hidden items-center gap-1.5 sm:inline-flex readout">
          <span className="live-dot inline-block h-[7px] w-[7px] rounded-full"
                style={{ background: "var(--bleached)", boxShadow: "0 0 0 4px var(--bleached-soft)" }} />
          Mock data
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}
