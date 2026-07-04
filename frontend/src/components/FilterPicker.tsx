import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";

import { fetchSpecies, type SpeciesEntry } from "../lib/api";
import { ChevronIcon } from "./FieldIcons";

/**
 * Feed filter. Mirrors SpeciesPicker's structure: special modes pinned at
 * top, then yard + broader NA species lists with a typeahead. Selecting a
 * row closes the modal.
 */
export type Filter =
  | { mode: "all" }
  | { mode: "interesting" } // hides junk feeder regulars (pigeons/doves/sparrows)
  | { mode: "unidentified" }
  | { mode: "awaiting_review" } // classifier-labeled, user hasn't acted yet
  | { mode: "nab" } // past 'Not a bird' labels — re-correct mistakes
  | { mode: "bad_quality" } // too dark / small / blurry
  | { mode: "binary_nab" } // crops the binary filter overrode to NAB (audit)
  | { mode: "species"; name: string };

type Props = {
  value: Filter;
  onChange: (next: Filter) => void;
  // "feed": everyday browsing filters + species typeahead.
  // "review": the niche audit cohorts (NAB / Bad crops / Binary NABs),
  // surfaced on the Review tab and kept OUT of the main feed filter.
  surface?: "feed" | "review";
};

export function filterLabel(f: Filter): string {
  switch (f.mode) {
    case "all":
      return "All birds";
    case "interesting":
      return "Interesting birds";
    case "unidentified":
      return "Unidentified only";
    case "awaiting_review":
      return "Awaiting review";
    case "nab":
      return "Not a bird";
    case "bad_quality":
      return "Bad crops";
    case "binary_nab":
      return "Binary-filter NABs";
    case "species":
      return f.name;
  }
}

type PinnedEntry = { filter: Filter; label: string; hint?: string };

// Everyday feed filters.
const FEED_PINNED: PinnedEntry[] = [
  { filter: { mode: "all" }, label: "All birds" },
  {
    filter: { mode: "interesting" },
    label: "Interesting birds",
    hint: "hides pigeons, doves & sparrows",
  },
  { filter: { mode: "unidentified" }, label: "Unidentified only", hint: "species_id IS NULL" },
];

// Review-tab cohorts. The active triage queues + audit views live here,
// pulled out of the main feed filter so day-to-day browsing isn't
// cluttered with maintenance views.
const REVIEW_PINNED: PinnedEntry[] = [
  {
    filter: { mode: "awaiting_review" },
    label: "Awaiting review",
    hint: "classifier labeled, not yet reviewed",
  },
  { filter: { mode: "nab" }, label: "Not a bird", hint: "past NAB labels — re-correct" },
  { filter: { mode: "bad_quality" }, label: "Bad crops", hint: "too small / dark / blurry" },
  {
    filter: { mode: "binary_nab" },
    label: "Binary-filter NABs",
    hint: "crops it killed — audit",
  },
];

export default function FilterPicker({ value, onChange, surface = "feed" }: Props) {
  const isReview = surface === "review";
  const pinned = isReview ? REVIEW_PINNED : FEED_PINNED;
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
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset-on-open runs once per open toggle, not on every render
      setQuery("");
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const { yardFiltered, extraFiltered } = useMemo(() => {
    if (!data) return { yardFiltered: [] as SpeciesEntry[], extraFiltered: [] as SpeciesEntry[] };
    const q = query.trim().toLowerCase();
    const match = (s: SpeciesEntry) => !q || s.name.toLowerCase().includes(q);
    // Sort the yard list by how many classified crops we have (desc) rather
    // than the backend's audio-frequency order — the species you've actually
    // captured most belong at the top of the feed filter.
    const yardByCrops = [...data.yard].sort((a, b) => (b.crops ?? 0) - (a.crops ?? 0));
    return {
      yardFiltered: yardByCrops.filter(match).slice(0, 50),
      extraFiltered: data.extra.filter(match).slice(0, q ? 50 : 100),
    };
  }, [data, query]);

  const pick = (next: Filter) => {
    onChange(next);
    setOpen(false);
  };

  const isActive = (f: Filter) =>
    f.mode === value.mode && (f.mode !== "species" || true);

  const rowBase =
    "w-full text-left px-3.5 py-2.5 text-sm border-b border-line/60 transition-colors hover:bg-[color-mix(in_oklab,var(--accent)_10%,transparent)]";
  const activeRow = "bg-[color-mix(in_oklab,var(--accent)_12%,transparent)] font-semibold text-ink";

  return (
    <>
      <button
        type="button"
        className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1.5 text-sm text-muted hover:border-leaf hover:text-leaf transition-colors"
        onClick={() => setOpen(true)}
        title="Filter the feed"
      >
        <span className="text-faint">Filter:</span>
        <span className="font-semibold text-ink">{filterLabel(value)}</span>
        <ChevronIcon size={14} className="text-faint" />
      </button>

      {open &&
        createPortal(
        <div
          className="fixed inset-0 z-50 flex items-start justify-center pt-16 backdrop-blur-sm bg-[color-mix(in_oklab,var(--ink)_52%,transparent)]"
          onClick={() => setOpen(false)}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="fg-card shadow-pop w-full max-w-md mx-4 max-h-[75vh] flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {!isReview && (
              <div className="p-3 border-b border-line">
                <input
                  ref={inputRef}
                  type="text"
                  placeholder="Search species…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="fg-input"
                  onKeyDown={(e) => {
                    if (e.key === "Escape") setOpen(false);
                  }}
                />
              </div>
            )}

            <div className="overflow-y-auto flex-1">
              {pinned.map((p) => (
                <button
                  key={p.label}
                  className={`${rowBase} flex items-center justify-between gap-2 ${
                    isActive(p.filter) && value.mode !== "species" ? activeRow : ""
                  }`}
                  onClick={() => pick(p.filter)}
                >
                  <span>{p.label}</span>
                  {p.hint && <span className="text-xs text-faint">{p.hint}</span>}
                </button>
              ))}

              {!isReview && (
                <>
              {isLoading && <p className="p-3 text-sm text-muted">Loading species…</p>}
              {!isLoading && yardFiltered.length === 0 && extraFiltered.length === 0 && query && (
                <p className="p-3 text-sm text-muted">No species match "{query}".</p>
              )}

              {yardFiltered.length > 0 && (
                <>
                  <div className="fg-overline px-3.5 pt-3 pb-1 bg-panel/40">
                    Heard in this yard
                  </div>
                  <ul>
                    {yardFiltered.map((s) => (
                      <li key={`yard-${s.name}`}>
                        <button
                          className={`${rowBase} flex items-center justify-between ${
                            value.mode === "species" && value.name === s.name ? activeRow : ""
                          }`}
                          onClick={() => pick({ mode: "species", name: s.name })}
                        >
                          <span>{s.name}</span>
                          {(s.crops ?? 0) > 0 && (
                            <span className="text-xs text-faint tnum" title="classified crops">
                              {(s.crops ?? 0) >= 1000
                                ? `${Math.round((s.crops ?? 0) / 1000)}k`
                                : s.crops}
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
                  <div className="fg-overline px-3.5 pt-3 pb-1 bg-panel/40 border-t border-line">
                    Other North American species
                  </div>
                  <ul>
                    {extraFiltered.map((s) => (
                      <li key={`extra-${s.name}`}>
                        <button
                          className={`${rowBase} ${
                            value.mode === "species" && value.name === s.name ? activeRow : ""
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
                </>
              )}
            </div>
          </div>
        </div>,
          document.body,
        )}
    </>
  );
}
