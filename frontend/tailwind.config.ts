import type { Config } from "tailwindcss";

const v = (name: string) => `var(--${name})`;

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: v("bg"),
        surface: v("surface"),
        "surface-2": v("surface-2"),
        "surface-inset": v("surface-inset"),
        ink: v("ink"),
        "ink-dim": v("ink-2s"),
        "ink-2s": v("ink-2s"),
        "ink-faint": v("ink-3"),
        "ink-3": v("ink-3"),
        line: v("line"),
        "line-2": v("line-2"),
        accent: v("accent"),
        "accent-ink": v("accent-ink"),
        "accent-soft": v("accent-soft"),
        healthy: v("healthy"),
        "healthy-fill": v("healthy-fill"),
        "healthy-soft": v("healthy-soft"),
        bleached: v("bleached"),
        "bleached-fill": v("bleached-fill"),
        "bleached-soft": v("bleached-soft"),
        uncertain: v("uncertain"),
        "uncertain-soft": v("uncertain-soft"),
        // legacy aliases
        abyss: v("bg"),
        deep: v("bg-2"),
        panel: v("surface"),
        cyan: v("accent"),
        flag: v("uncertain"),
      },
      fontFamily: {
        display: [v("font-display"), "Georgia", "serif"],
        mono: [v("font-mono"), "ui-monospace", "monospace"],
        body: [v("font-body"), "system-ui", "sans-serif"],
      },
      borderRadius: { lg: v("r-lg"), md: v("r-md"), sm: v("r-sm") },
      boxShadow: { soft: v("shadow"), lift: v("shadow-lift") },
      letterSpacing: { readout: "0.18em" },
    },
  },
  plugins: [],
};
export default config;
