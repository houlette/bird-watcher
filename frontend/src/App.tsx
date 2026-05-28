import { NavLink, Route, Routes } from "react-router-dom";

import Feed from "./pages/Feed";
import Settings from "./pages/Settings";
import Species from "./pages/Species";

export default function App() {
  return (
    <div className="min-h-full flex flex-col">
      <header className="bg-forest text-cream px-4 py-3 shadow">
        <h1 className="text-xl font-semibold">BirdWatcher</h1>
      </header>

      <main className="flex-1 max-w-3xl w-full mx-auto p-4">
        <Routes>
          <Route path="/" element={<Feed />} />
          <Route path="/labels" element={<Feed mode="nab" />} />
          <Route path="/species/:id" element={<Species />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>

      <nav className="bg-white border-t flex justify-around py-2 sticky bottom-0">
        <Tab to="/" label="Feed" />
        <Tab to="/labels" label="Labels" />
        <Tab to="/settings" label="Settings" />
      </nav>
    </div>
  );
}

function Tab({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      end
      className={({ isActive }) =>
        `px-4 py-1 rounded ${isActive ? "text-forest font-semibold" : "text-slate-500"}`
      }
    >
      {label}
    </NavLink>
  );
}
