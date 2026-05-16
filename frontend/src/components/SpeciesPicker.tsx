import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchSpecies, type SpeciesEntry } from "../lib/api";

type Props = {
  open: boolean;
  current?: string | null;
  onSelect: (species: string) => void;
  onCancel: () => void;
};

/**
 * Searchable combobox over the yard's allow-listed species (~157 entries when
 * calibration is loaded, ~60 from the hand-coded fallback otherwise). Drives
 * the "Wrong species?" UX on DetectionCard.
 */
export default function SpeciesPicker({ open, current, onSelect, onCancel }: Props) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  const { data, isLoading } = useQuery({
    queryKey: ["species"],
    queryFn: fetchSpecies,
    staleTime: 5 * 60_000,
  });

  useEffect(() => {
    if (open) {
      setQuery("");
      // Defer focus so the modal has mounted.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const filtered = useMemo<SpeciesEntry[]>(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    if (!q) return data.species.slice(0, 50); // show top 50 by count when no query
    return data.species.filter((s) => s.name.toLowerCase().includes(q)).slice(0, 50);
  }, [data, query]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 bg-black/40 z-50 flex items-start justify-center pt-20"
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4 max-h-[70vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-3 border-b">
          <input
            ref={inputRef}
            type="text"
            placeholder="Search species…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 rounded outline-none focus:border-forest"
            onKeyDown={(e) => {
              if (e.key === "Escape") onCancel();
              if (e.key === "Enter" && filtered.length > 0) onSelect(filtered[0].name);
            }}
          />
        </div>

        <div className="overflow-y-auto flex-1">
          {isLoading && <p className="p-3 text-sm text-slate-500">Loading species…</p>}
          {!isLoading && filtered.length === 0 && (
            <p className="p-3 text-sm text-slate-500">No species match "{query}".</p>
          )}
          <ul>
            {filtered.map((s) => (
              <li key={s.name}>
                <button
                  className={`w-full text-left px-3 py-2 hover:bg-slate-50 flex items-center justify-between text-sm ${
                    s.name === current ? "bg-cream font-semibold" : ""
                  }`}
                  onClick={() => onSelect(s.name)}
                >
                  <span>{s.name}</span>
                  {s.total > 0 && (
                    <span className="text-xs text-slate-400">
                      {s.total >= 1000 ? `${Math.round(s.total / 1000)}k` : s.total}
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="p-2 border-t flex justify-between text-xs text-slate-500">
          <span>
            {data?.source === "calibration"
              ? "Species from this yard's Haikubox history."
              : "Fallback species list (calibrate for better picks)."}
          </span>
          <button onClick={onCancel} className="text-slate-600 hover:text-slate-900">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
