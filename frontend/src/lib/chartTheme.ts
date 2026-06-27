// ─── Field-guide chart palette ────────────────────────────────────────────
//
// Recharts wants raw CSS colours (it can't read CSS variables), so we keep a
// light/dark categorical palette here and pick by the active theme. Greens &
// rust come straight from the design tokens; the blue/sand/teal are tuned to
// sit harmoniously beside sage in both themes while staying mutually
// distinguishable on a multi-series chart. Shared by the Stats and Insights
// pages so the two surfaces read as one field guide.
export type Tokens = {
  leaf: string;
  rust: string;
  blue: string;
  sand: string;
  slate: string;
  teal: string;
  band: string;
  ink: string;
  axis: string;
  grid: string;
  tipBg: string;
};

const LIGHT: Tokens = {
  leaf: "#356544",
  rust: "#b0552f",
  blue: "#41698c",
  sand: "#bc8a3e",
  slate: "#8a8472",
  teal: "#2a7d72",
  band: "rgba(42,125,114,0.16)",
  ink: "#212a1e",
  axis: "#86927b",
  grid: "#d2dcc4",
  tipBg: "#f3f5ec",
};

const DARK: Tokens = {
  leaf: "#84ba90",
  rust: "#d68a4f",
  blue: "#8fb4cf",
  sand: "#d8b877",
  slate: "#9aa68f",
  teal: "#74c2b4",
  band: "rgba(116,194,180,0.18)",
  ink: "#e9eddf",
  axis: "#74806b",
  grid: "#2f3829",
  tipBg: "#232b1f",
};

export function readTokens(): Tokens {
  const dark =
    typeof document !== "undefined" &&
    document.documentElement.classList.contains("dark");
  return dark ? DARK : LIGHT;
}

// Shared Tooltip styling so popovers match the surface in both themes.
export function tip(t: Tokens) {
  return {
    contentStyle: {
      background: t.tipBg,
      border: `1px solid ${t.grid}`,
      borderRadius: 10,
      fontSize: 12,
      color: t.ink,
      boxShadow: "0 14px 30px -16px rgba(24,26,18,.35)",
    },
    labelStyle: { color: t.ink, fontWeight: 600 },
    itemStyle: { color: t.ink },
  };
}
