/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  // Enable the "Twilight" dark theme by toggling class="dark" on <html>.
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // ── Field-guide tokens ────────────────────────────────────────
        // Every value is a CSS variable defined in src/index.css, so the
        // single source of truth for the light/dark palettes lives in one
        // place and Tailwind's `/<alpha>` opacity modifiers still work via
        // <alpha-value>.
        paper: "var(--bg)", // page background
        panel: "var(--panel)", // recessed surfaces (toolbar, nav)
        surface: "var(--card)", // cards, modals
        ink: "var(--ink)", // primary text
        muted: "var(--muted)", // secondary text
        faint: "var(--faint)", // tertiary text / hairline labels
        line: "var(--line)", // borders / rules
        leaf: "var(--accent)", // brand accent (actions, active state)
        rust: "var(--rust)", // destructive / warning

        // Back-compat aliases so any un-migrated class still resolves to a
        // field-guide token instead of the old hard-coded hex.
        forest: "var(--accent)",
        cream: "var(--card)",
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
