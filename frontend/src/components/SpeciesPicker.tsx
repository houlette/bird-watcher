import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchSpecies, type SpeciesEntry } from "../lib/api";

// Must match backend/db/models.py:NOT_A_BIRD_LABEL exactly. The picker
// pins this as a special option (separated from real species by a divider)
// so users can flag false positives like the hummingbird feeder without
// having to scroll through 150 real species names.
export const NOT_A_BIRD = "Not a bird";

// Must match backend/db/models.py:UNKNOWN_BIRD_LABEL. Pinned alongside
// "Not a bird" for the case where the user is sure it IS a bird but
// can't ID the species — still a useful positive training label for the
// YOLO detector even without species info.
export const UNKNOWN_BIRD = "Unknown bird";

type Props = {
  open: boolean;
  current?: string | null;
  onSelect: (species: string) => void;
  onCancel: () => void;
};

/**
 * Searchable combobox over species labels. Renders two groups:
 *   1. Yard species — what the Haikubox has actually heard at this yard,
 *      sorted by detection count, with the counts shown as hints.
 *   2. Other NA species — a broader curated NA-bird list (pigeons,
 *      raptors, etc.) for species the user sees but the Haikubox has
 *      never recorded. Alphabetical.
 *
 * When the user types a query, both groups are filtered and rendered
 * under the same query result.
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
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Filter both groups by query. When no query, show all yard species
  // and the top of the alphabetical extra list (capped so the picker
  // doesn't render hundreds of items at once).
  const { yardFiltered, extraFiltered } = useMemo(() => {
    if (!data) return { yardFiltered: [] as SpeciesEntry[], extraFiltered: [] as SpeciesEntry[] };
    const q = query.trim().toLowerCase();
    const match = (s: SpeciesEntry) => !q || s.name.toLowerCase().includes(q);
    const yardFiltered = data.yard.filter(match).slice(0, 50);
    const extraFiltered = data.extra.filter(match).slice(0, q ? 50 : 100);
    return { yardFiltered, extraFiltered };
  }, [data, query]);

  // Enter key picks the first available match across both groups.
  const firstMatch = yardFiltered[0]?.name ?? extraFiltered[0]?.name;

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
              if (e.key === "Enter" && firstMatch) onSelect(firstMatch);
            }}
          />
        </div>

        <div className="overflow-y-auto flex-1">
          {/* Pinned sentinel options — always at the top, distinct styling.
              Shown even with an active query so users can flag false positives
              or species-unknown birds without typing. */}
          <button
            className="w-full text-left px-3 py-2 hover:bg-slate-50 flex items-center justify-between text-sm border-b border-slate-100 text-slate-600"
            onClick={() => onSelect(NOT_A_BIRD)}
          >
            <span className="flex items-center gap-2">
              <span aria-hidden>🚫</span>
              <span>{NOT_A_BIRD}</span>
            </span>
            <span className="text-xs text-slate-400">false positive</span>
          </button>
          <button
            className="w-full text-left px-3 py-2 hover:bg-slate-50 flex items-center justify-between text-sm border-b border-slate-200 text-slate-600"
            onClick={() => onSelect(UNKNOWN_BIRD)}
          >
            <span className="flex items-center gap-2">
              <span aria-hidden>❓</span>
              <span>{UNKNOWN_BIRD}</span>
            </span>
            <span className="text-xs text-slate-400">bird, species unsure</span>
          </button>

          {isLoading && <p className="p-3 text-sm text-slate-500">Loading species…</p>}
          {!isLoading && yardFiltered.length === 0 && extraFiltered.length === 0 && (
            <p className="p-3 text-sm text-slate-500">No species match "{query}".</p>
          )}

          {/* Group 1: Yard species (Haikubox-heard, with detection counts). */}
          {yardFiltered.length > 0 && (
            <>
              <div className="px-3 pt-2 pb-1 text-[10px] uppercase tracking-wide text-slate-400 bg-slate-50/50">
                Heard in this yard
              </div>
              <ul>
                {yardFiltered.map((s) => (
                  <li key={`yard-${s.name}`}>
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
            </>
          )}

          {/* Group 2: Broader NA bird list. */}
          {extraFiltered.length > 0 && (
            <>
              <div className="px-3 pt-2 pb-1 text-[10px] uppercase tracking-wide text-slate-400 bg-slate-50/50 border-t border-slate-100">
                Other North American species
              </div>
              <ul>
                {extraFiltered.map((s) => (
                  <li key={`extra-${s.name}`}>
                    <button
                      className={`w-full text-left px-3 py-2 hover:bg-slate-50 text-sm ${
                        s.name === current ? "bg-cream font-semibold" : ""
                      }`}
                      onClick={() => onSelect(s.name)}
                    >
                      {s.name}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>

        <div className="p-2 border-t flex justify-between text-xs text-slate-500">
          <span>
            {data?.source === "calibration"
              ? `${data.yard.length} from yard · ${data.extra.length} other NA species`
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
