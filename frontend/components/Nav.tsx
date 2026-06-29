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
  const live = !!process.env.NEXT_PUBLIC_REEFSCAN_API;

  return (
    <header className="mx-auto flex w-full max-w-[1240px] flex-wrap items-center justify-between gap-4 px-4 py-5 md:px-8">
      <Link href="/" className="flex items-center gap-3" style={{ textDecoration: "none", color: "inherit" }}>
        <span className="grid h-[38px] w-[38px] place-items-center overflow-hidden rounded-xl"
              style={{ background: "#fff", boxShadow: "var(--shadow)", border: "1px solid var(--line)" }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="ReefScan logo" width={28} height={28} />
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
        <span className="hidden items-center gap-1.5 sm:inline-flex readout" title={live ? "Connected to the live inference API" : "Using built-in mock data"}>
          <span className="live-dot inline-block h-[7px] w-[7px] rounded-full"
                style={{ background: live ? "var(--healthy)" : "var(--bleached)",
                         boxShadow: `0 0 0 4px ${live ? "var(--healthy-soft)" : "var(--bleached-soft)"}` }} />
          {live ? "Live API" : "Mock data"}
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}
