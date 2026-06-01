import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchSpecies, type SpeciesEntry } from "../lib/api";

/**
 * Feed filter. Three states:
 *   - `{ mode: "all" }`             show every detection
 *   - `{ mode: "unidentified" }`    only species_id IS NULL
 *   - `{ mode: "species", name }`   only this species (by common name)
 *
 * The picker is a small modal that mirrors SpeciesPicker's structure: the
 * two special modes pinned at top, then yard + broader NA species lists
 * with a typeahead. Selecting a row closes the modal.
 */
export type Filter =
  | { mode: "all" }
  | { mode: "unidentified" }
  | { mode: "llm_review" }
  | { mode: "species"; name: string };

type Props = {
  value: Filter;
  onChange: (next: Filter) => void;
};

export function filterLabel(f: Filter): string {
  switch (f.mode) {
    case "all":
      return "All birds";
    case "unidentified":
      return "Unidentified only";
    case "llm_review":
      return "LLM-labeled (review)";
    case "species":
      return f.name;
  }
}

export default function FilterPicker({ value, onChange }: Props) {
  const [open, setOpen] = useState(false);
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

  const { yardFiltered, extraFiltered } = useMemo(() => {
    if (!data) return { yardFiltered: [] as SpeciesEntry[], extraFiltered: [] as SpeciesEntry[] };
    const q = query.trim().toLowerCase();
    const match = (s: SpeciesEntry) => !q || s.name.toLowerCase().includes(q);
    return {
      yardFiltered: data.yard.filter(match).slice(0, 50),
      extraFiltered: data.extra.filter(match).slice(0, q ? 50 : 100),
    };
  }, [data, query]);

  const pick = (next: Filter) => {
    onChange(next);
    setOpen(false);
  };

  return (
    <>
      <button
        type="button"
        className="px-3 py-1 text-sm rounded border bg-white text-slate-700 border-slate-200 hover:border-forest hover:text-forest inline-flex items-center gap-1"
        onClick={() => setOpen(true)}
        title="Filter the feed"
      >
        <span className="text-slate-400">Filter:</span>
        <span className="font-medium">{filterLabel(value)}</span>
        <span className="text-slate-400">▾</span>
      </button>

      {open && (
        <div
          className="fixed inset-0 bg-black/40 z-50 flex items-start justify-center pt-16"
          onClick={() => setOpen(false)}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4 max-h-[75vh] flex flex-col"
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
                  if (e.key === "Escape") setOpen(false);
                }}
              />
            </div>

            <div className="overflow-y-auto flex-1">
              {/* Pinned: All birds / Unidentified — always shown above the
                  species list so the user can jump back to the default view
                  or to the "needs labeling" queue without scrolling. */}
              <button
                className={`w-full text-left px-3 py-2 hover:bg-slate-50 text-sm border-b border-slate-100 ${
                  value.mode === "all" ? "bg-cream font-semibold" : ""
                }`}
                onClick={() => pick({ mode: "all" })}
              >
                All birds
              </button>
              <button
                className={`w-full text-left px-3 py-2 hover:bg-slate-50 text-sm border-b border-slate-100 ${
                  value.mode === "unidentified" ? "bg-cream font-semibold" : ""
                }`}
                onClick={() => pick({ mode: "unidentified" })}
              >
                ❓ Unidentified only
              </button>
              <button
                className={`w-full text-left px-3 py-2 hover:bg-slate-50 text-sm border-b border-slate-200 ${
                  value.mode === "llm_review" ? "bg-cream font-semibold" : ""
                }`}
                onClick={() => pick({ mode: "llm_review" })}
                title="Detections labeled by the LLM backlog-classifier pass — review and correct any wrong ones."
              >
                ✨ LLM-labeled (review)
              </button>

              {isLoading && <p className="p-3 text-sm text-slate-500">Loading species…</p>}
              {!isLoading && yardFiltered.length === 0 && extraFiltered.length === 0 && query && (
                <p className="p-3 text-sm text-slate-500">No species match "{query}".</p>
              )}

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
                            value.mode === "species" && value.name === s.name ? "bg-cream font-semibold" : ""
                          }`}
                          onClick={() => pick({ mode: "species", name: s.name })}
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
                            value.mode === "species" && value.name === s.name ? "bg-cream font-semibold" : ""
                          }`}
                          onClick={() => pick({ mode: "species", name: s.name })}
                        >
                          {s.name}
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
