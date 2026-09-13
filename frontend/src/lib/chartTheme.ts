import { useSyncExternalStore } from "react";

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
  // The card colour. Doubles as the tooltip background and as the gap
  // colour between touching marks — a stacked column separates its
  // segments with a 2px stroke in the surface, not a border.
  surface: string;
  // Ordinal ramps, palest → deepest. `funnel` colours pipeline depth on
  // the Stats funnel (palest = dropped first, darkest = got furthest);
  // `removed` colours the three spatial defences, which only ever take
  // things away, so they wear the rust counter-colour instead.
  //
  // Both were generated in OKLCH off the --accent / --rust tokens and
  // checked with the dataviz validator in ordinal mode (monotone
  // lightness, adjacent ΔL ≥ 0.06, one hue, palest step ≥ 2:1 against
  // that mode's surface). Retune light and dark together, and re-run the
  // validator — the dark steps are chosen against the dark card, not
  // flipped from the light ones.
  funnel: [string, string, string, string];
  removed: [string, string, string];
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
  surface: "#f3f5ec",
  funnel: ["#8fb097", "#659472", "#3b774f", "#255836"],
  removed: ["#c99c8a", "#bf7a5f", "#b15732"],
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
  surface: "#232b1f",
  funnel: ["#59705d", "#668f6f", "#77ae83", "#8acd9a"],
  removed: ["#906c52", "#be8356", "#ed9b5d"],
};

const isDark = () =>
  typeof document !== "undefined" &&
  document.documentElement.classList.contains("dark");

function subscribeToTheme(onChange: () => void): () => void {
  if (typeof MutationObserver === "undefined") return () => {};
  const obs = new MutationObserver(onChange);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  return () => obs.disconnect();
}

/** The active palette, re-read when the theme changes.
 *
 *  The theme toggle flips a class on <html>, which React knows nothing
 *  about, so a chart that read the palette during render kept the one it
 *  mounted with until the next reload — light-mode fills and light-mode
 *  gaps on a dark card. Watching the class keeps the fills, the axes and
 *  the surface-coloured gaps between stacked segments in step with it. */
export function useTokens(): Tokens {
  return useSyncExternalStore(
    subscribeToTheme,
    () => (isDark() ? DARK : LIGHT),
    // Server snapshot: no document to read, so assume light.
    () => LIGHT,
  );
}

// Shared Tooltip styling so popovers match the surface in both themes.
export function tip(t: Tokens) {
  return {
    contentStyle: {
      background: t.surface,
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
