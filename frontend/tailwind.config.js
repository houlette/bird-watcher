/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  // Enable the "Twilight" dark theme by toggling class="dark" on <html>.
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // ── Field-guide tokens ────────────────────────────────────────
        // Each token reads an RGB-triplet CSS variable from src/index.css
        // (the single source of truth for the light/dark palettes). The
        // `rgb(... / <alpha-value>)` form is REQUIRED for Tailwind's
        // `/<alpha>` opacity modifiers: with a plain `var(--card)` hex,
        // `bg-surface/85` compiles to `rgb(#232b1f / 0.85)` — invalid CSS
        // the browser silently drops (transparent chips, default-blue
        // rings).
        paper: "rgb(var(--bg-rgb) / <alpha-value>)", // page background
        panel: "rgb(var(--panel-rgb) / <alpha-value>)", // recessed surfaces (toolbar, nav)
        surface: "rgb(var(--card-rgb) / <alpha-value>)", // cards, modals
        ink: "rgb(var(--ink-rgb) / <alpha-value>)", // primary text
        muted: "rgb(var(--muted-rgb) / <alpha-value>)", // secondary text
        faint: "rgb(var(--faint-rgb) / <alpha-value>)", // tertiary text / hairline labels
        line: "rgb(var(--line-rgb) / <alpha-value>)", // borders / rules
        leaf: "rgb(var(--accent-rgb) / <alpha-value>)", // brand accent (actions, active state)
        rust: "rgb(var(--rust-rgb) / <alpha-value>)", // destructive / warning

        // Back-compat aliases so any un-migrated class still resolves to a
        // field-guide token instead of the old hard-coded hex.
        forest: "rgb(var(--accent-rgb) / <alpha-value>)",
        cream: "rgb(var(--card-rgb) / <alpha-value>)",
      },
      fontFamily: {
        serif: ['"Newsreader"', "Georgia", "serif"],
        sans: ['"Hanken Grotesk"', "system-ui", "sans-serif"],
      },
      borderRadius: {
        card: "var(--radius)",
      },
      boxShadow: {
        plate: "0 1px 2px rgba(20,22,16,.04)",
        lift: "0 14px 30px -16px rgba(24,26,18,.35)",
        pop: "0 30px 80px -30px rgba(0,0,0,.6)",
      },
    },
  },
  plugins: [],
};
