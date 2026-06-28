import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        abyss: "var(--bg-abyss)",
        deep: "var(--bg-deep)",
        panel: "var(--bg-panel)",
        ink: "var(--ink)",
        "ink-dim": "var(--ink-dim)",
        "ink-faint": "var(--ink-faint)",
        line: "var(--line)",
        healthy: "var(--healthy)",
        bleached: "var(--bleached)",
        flag: "var(--flag)",
        cyan: "var(--cyan)",
      },
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
        body: ["var(--font-body)", "system-ui", "sans-serif"],
      },
      letterSpacing: {
        readout: "0.18em",
      },
    },
  },
  plugins: [],
};
export default config;
