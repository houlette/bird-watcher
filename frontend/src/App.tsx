import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";

import Feed from "./pages/Feed";
import Settings from "./pages/Settings";
import Species from "./pages/Species";
import Stats from "./pages/Stats";
import {
  BirdMark,
  ChartIcon,
  FeedIcon,
  GearIcon,
  MoonIcon,
  SunIcon,
  TagIcon,
} from "./components/FieldIcons";

/**
 * Light/dark ("Sage"/"Twilight") theme toggle. The class is applied to
 * <html> (see index.html's pre-mount script that restores it) and the
 * choice is persisted so it survives reloads.
 */
function useTheme(): [boolean, () => void] {
  const [dark, setDark] = useState(
    () =>
      typeof document !== "undefined" &&
      document.documentElement.classList.contains("dark"),
  );
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("bw-theme", dark ? "dark" : "light");
    } catch {
      /* private mode — ignore */
    }
  }, [dark]);
  return [dark, () => setDark((d) => !d)];
}

export default function App() {
  const [dark, toggleTheme] = useTheme();

  return (
    <div className="min-h-full flex flex-col bg-paper text-ink">
      {/* ── Masthead ───────────────────────────────────────────────────── */}
      <header className="max-w-3xl w-full mx-auto px-4 pt-5">
        {/* No flex-wrap: on a 375px phone the status block used to wrap
            under the title, doubling the masthead height. The "Watching"
            word hides on xs (dot + tooltip carry the meaning) so one row
            always fits. */}
        <div className="flex items-end justify-between gap-4">
          <div className="flex items-center gap-3.5 min-w-0">
            <span className="grid place-items-center w-11 h-11 shrink-0 rounded-full text-leaf border border-line bg-[color-mix(in_oklab,var(--accent)_12%,var(--card))]">
              <BirdMark size={24} />
            </span>
            <div className="min-w-0">
              <div className="fg-overline truncate">Backyard Feeder Station</div>
              <h1 className="font-serif font-medium text-ink leading-none tracking-tight text-3xl mt-0.5">
                BirdWatcher
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-3 pb-1 shrink-0">
            <span
              className="inline-flex items-center gap-2 text-leaf text-[11px] font-semibold uppercase tracking-[0.14em]"
              title="Pipeline live — watching for new clips"
            >
              <i className="fg-livedot" /> <span className="hidden sm:inline">Watching</span>
            </span>
            <button
              onClick={toggleTheme}
              className="grid place-items-center w-9 h-9 rounded-full border border-line text-muted hover:text-leaf hover:border-leaf transition-colors"
              aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
              title={dark ? "Sage (light)" : "Twilight (dark)"}
            >
              {dark ? <SunIcon size={18} /> : <MoonIcon size={18} />}
            </button>
          </div>
        </div>
        <div className="fg-rule mt-4" aria-hidden />
      </header>

      <main className="flex-1 max-w-3xl w-full mx-auto px-4 pt-4 pb-28">
        <Routes>
          <Route path="/" element={<Feed />} />
          <Route path="/labels" element={<Feed mode="nab" />} />
          <Route path="/species/:id" element={<Species />} />
          <Route path="/stats" element={<Stats />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>

      {/* ── Bottom nav ─────────────────────────────────────────────────── */}
      <nav className="sticky bottom-0 z-30 max-w-[560px] w-full mx-auto flex justify-around gap-1 px-2 pt-2 pb-[calc(0.5rem+env(safe-area-inset-bottom,0px))] bg-[color-mix(in_oklab,var(--panel)_92%,transparent)] backdrop-blur-md border border-line border-b-0 rounded-t-2xl">
        <Tab to="/" label="Feed" icon={<FeedIcon size={20} />} />
        <Tab to="/labels" label="Labels" icon={<TagIcon size={20} />} />
        <Tab to="/stats" label="Stats" icon={<ChartIcon size={20} />} />
        <Tab to="/settings" label="Settings" icon={<GearIcon size={20} />} />
      </nav>
    </div>
  );
}

function Tab({
  to,
  label,
  icon,
}: {
  to: string;
  label: string;
  icon: React.ReactNode;
}) {
  return (
    <NavLink
      to={to}
      end
      className={({ isActive }) =>
        `flex flex-col items-center gap-0.5 px-3.5 py-1 rounded-lg text-[10.5px] font-semibold tracking-wide transition-colors ${
          isActive ? "text-leaf" : "text-faint hover:text-muted"
        }`
      }
    >
      {icon}
      <span>{label}</span>
    </NavLink>
  );
}
