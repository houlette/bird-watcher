import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchSpecies, type FamilyEntry, type SpeciesEntry } from "../lib/api";

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

/**
 * Optional list of the classifier's top-K predictions surfaced as a
 * "Suggested" group at the top of the picker. When the user taps
 * "Wrong species?" we know the top-1 was wrong, so we exclude it and
 * show what the classifier ranked #2 through #5 — usually one of them
 * is the correct ID, and a single tap finishes the correction instead
 * of searching the full list.
 */
export type Suggestion = { species: string; p: number };

type Props = {
  open: boolean;
  current?: string | null;
  suggestions?: Suggestion[];
  // Optional crop URL for side-by-side compare. When provided, the
  // picker shows the user's crop pinned at the top of the modal next to
  // the reference photo of whichever species the user is hovering, so
  // they can do a real A/B visual compare without closing the picker.
  cropUrl?: string;
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
export default function SpeciesPicker({ open, current, suggestions, cropUrl, onSelect, onCancel }: Props) {
  // Filter out the current species (the wrong top-1 that prompted the
  // correction) and cap at the next 3 ranked guesses.
  const suggestionList = (suggestions ?? [])
    .filter((s) => s.species && s.species !== current)
    .slice(0, 3);
  const [query, setQuery] = useState("");
  // Which species the user is currently hovering / keyboard-focusing.
  // Drives the right pane of the compare strip so they can visually
  // A/B their crop against any candidate without closing the picker.
  const [hovered, setHovered] = useState<{ name: string; url: string | null } | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const { data, isLoading } = useQuery({
    queryKey: ["species"],
    queryFn: fetchSpecies,
    staleTime: 5 * 60_000,
  });

  useEffect(() => {
    if (open) {
      setQuery("");
      setHovered(null);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Filter both groups by query. When no query, show all yard species
  // and the top of the alphabetical extra list (capped so the picker
  // doesn't render hundreds of items at once).
  const { familiesFiltered, yardFiltered, extraFiltered } = useMemo(() => {
    const empty = {
      familiesFiltered: [] as FamilyEntry[],
      yardFiltered: [] as SpeciesEntry[],
      extraFiltered: [] as SpeciesEntry[],
    };
    if (!data) return empty;
    const q = query.trim().toLowerCase();
    const matchSp = (s: SpeciesEntry) => !q || s.name.toLowerCase().includes(q);
    // Family matches against its own name AND its member-species names so
    // typing "junco" surfaces Sparrow.
    const matchFam = (f: FamilyEntry) =>
      !q ||
      f.name.toLowerCase().includes(q) ||
      f.members.some((m) => m.toLowerCase().includes(q));
    return {
      familiesFiltered: (data.families ?? []).filter(matchFam),
      yardFiltered: data.yard.filter(matchSp).slice(0, 50),
      extraFiltered: data.extra.filter(matchSp).slice(0, q ? 50 : 100),
    };
  }, [data, query]);

  // Enter key picks the first available match across all groups
  // (families take precedence — typing "sparrow" + Enter picks the
  // family, not the first species).
  const firstMatch =
    familiesFiltered[0]?.name ?? yardFiltered[0]?.name ?? extraFiltered[0]?.name;

  // Reference-URL lookup across all species/families. Built once per
  // data load so per-row hover handlers can resolve a URL in O(1) for
  // suggestion-list entries (where we don't have the full row inline).
  const refUrlByName = useMemo(() => {
    const m = new Map<string, string | null>();
    if (!data) return m;
    for (const f of data.families ?? []) m.set(f.name, f.reference_image_url ?? null);
    for (const s of data.yard) m.set(s.name, s.reference_image_url ?? null);
    for (const s of data.extra) m.set(s.name, s.reference_image_url ?? null);
    return m;
  }, [data]);
  const currentRefUrl = current ? refUrlByName.get(current) ?? null : null;

  // Helper to build hover/focus handlers for every selectable row.
  // Keeps the JSX below readable instead of repeating 4 inline handlers
  // per group. We treat keyboard focus the same as mouse hover so
  // tab-through navigation also drives the compare pane.
  const hoverProps = (name: string, url: string | null | undefined) => ({
    onMouseEnter: () => setHovered({ name, url: url ?? null }),
    onFocus: () => setHovered({ name, url: url ?? null }),
  });

  // Right pane defaults to current species when nothing is hovered.
  const comparePane = hovered ?? (current ? { name: current, url: currentRefUrl } : null);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 bg-black/40 z-50 flex items-start justify-center pt-20"
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="bg-white rounded-lg shadow-xl w-full max-w-lg mx-4 max-h-[80vh] flex flex-col"
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

        {/* Compare strip: user's crop on the left, hovered/current species
            reference photo on the right. Pinned above the scrolling list so
            both stay visible while the user scans candidates. Hidden when
            no crop URL was supplied (caller didn't opt in to compare mode). */}
        {cropUrl && (
          <div className="grid grid-cols-2 gap-px bg-slate-200 border-b border-slate-200">
            <div className="relative bg-slate-900 aspect-[4/3] overflow-hidden">
              <img
                src={cropUrl}
                alt="your crop"
                className="absolute inset-0 w-full h-full object-contain"
              />
              <span className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-white/85 text-[10px] font-medium text-slate-600 uppercase tracking-wide">
                your crop
              </span>
            </div>
            <div className="relative bg-slate-900 aspect-[4/3] overflow-hidden">
              {comparePane && comparePane.url ? (
                <>
                  <img
                    src={comparePane.url}
                    aria-hidden
                    className="absolute inset-0 w-full h-full object-cover blur-2xl scale-110 opacity-70"
                  />
                  <img
                    src={comparePane.url}
                    alt={comparePane.name}
                    className="relative w-full h-full object-contain"
                  />
                  <span className="absolute top-1 left-1 right-1 px-1.5 py-0.5 rounded bg-white/85 text-[10px] font-medium text-slate-700 truncate">
                    {comparePane.name}
                  </span>
                </>
              ) : (
                <div className="absolute inset-0 flex items-center justify-center text-[11px] text-slate-400 text-center px-3">
                  {comparePane
                    ? `${comparePane.name}\n(no reference photo)`
                    : "Hover a species to compare"}
                </div>
              )}
            </div>
          </div>
        )}

        <div
          className="overflow-y-auto flex-1"
          // Once the mouse leaves the list (back to search box, outside
          // the modal, etc.), revert the compare strip's right pane to
          // the current classification rather than leaving it stuck on
          // whatever row was last hovered.
          onMouseLeave={() => setHovered(null)}
        >
          {/* Pinned sentinel options — always at the top, distinct styling.
              Shown even with an active query so users can flag false positives
              or species-unknown birds without typing. */}
          <button
            className="w-full text-left px-3 py-2 hover:bg-slate-50 flex items-center justify-between text-sm border-b border-slate-100 text-slate-600"
            onClick={() => onSelect(NOT_A_BIRD)}
            {...hoverProps(NOT_A_BIRD, null)}
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
            {...hoverProps(UNKNOWN_BIRD, null)}
          >
            <span className="flex items-center gap-2">
              <span aria-hidden>❓</span>
              <span>{UNKNOWN_BIRD}</span>
            </span>
            <span className="text-xs text-slate-400">bird, species unsure</span>
          </button>

          {/* Classifier's next-best guesses for this crop (top-1 excluded — it's
              what the user just rejected). If the right answer is in here, the
              correction is a single tap. Only shown when no search query is
              active so the suggestions don't get hidden behind a filtered list. */}
          {suggestionList.length > 0 && !query && (
            <>
              <div className="px-3 pt-2 pb-1 text-[10px] uppercase tracking-wide text-slate-400 bg-slate-50/50 border-t border-slate-100">
                Suggested by classifier
              </div>
              <ul>
                {suggestionList.map((s) => (
                  <li key={`suggestion-${s.species}`}>
                    <button
                      className="w-full text-left px-3 py-2 hover:bg-slate-50 flex items-center justify-between text-sm border-b border-slate-50"
                      onClick={() => onSelect(s.species)}
                      {...hoverProps(s.species, refUrlByName.get(s.species) ?? null)}
                    >
                      <span className="flex items-center gap-2">
                        <span aria-hidden>✨</span>
                        <span>{s.species}</span>
                      </span>
                      <span className="text-xs text-slate-400">
                        {Math.round(s.p * 100)}%
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}

          {isLoading && <p className="p-3 text-sm text-slate-500">Loading species…</p>}
          {!isLoading &&
            familiesFiltered.length === 0 &&
            yardFiltered.length === 0 &&
            extraFiltered.length === 0 && (
              <p className="p-3 text-sm text-slate-500">No species match "{query}".</p>
            )}

          {/* Family-level catch-alls — "I know it's a sparrow but I can't
              tell which kind." Distinct group so the picker visually
              cues that these are a different kind of label. Members
              shown as a hint so the user remembers what each covers. */}
          {familiesFiltered.length > 0 && (
            <>
              <div className="px-3 pt-2 pb-1 text-[10px] uppercase tracking-wide text-slate-400 bg-slate-50/50 border-t border-slate-100">
                Families
              </div>
              <ul>
                {familiesFiltered.map((f) => (
                  <li key={`family-${f.name}`}>
                    <button
                      className={`w-full text-left px-3 py-2 hover:bg-slate-50 text-sm flex items-start gap-2 ${
                        f.name === current ? "bg-cream font-semibold" : ""
                      }`}
                      onClick={() => onSelect(f.name)}
                      {...hoverProps(f.name, f.reference_image_url)}
                    >
                      <SpeciesThumb url={f.reference_image_url} fallback="👥" />
                      <span className="flex-1 min-w-0">
                        <span className="block">{f.name}</span>
                        <span className="block text-xs text-slate-400 truncate">
                          e.g., {f.members.slice(0, 3).join(", ")}
                          {f.members.length > 3 ? ", …" : ""}
                        </span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
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
                      className={`w-full text-left px-3 py-2 hover:bg-slate-50 text-sm flex items-center gap-2 ${
                        s.name === current ? "bg-cream font-semibold" : ""
                      }`}
                      onClick={() => onSelect(s.name)}
                      {...hoverProps(s.name, s.reference_image_url)}
                    >
                      <SpeciesThumb url={s.reference_image_url} />
                      <span className="flex-1 min-w-0">{s.name}</span>
                      {s.total > 0 && (
                        <span className="text-xs text-slate-400 flex-shrink-0">
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
                      className={`w-full text-left px-3 py-2 hover:bg-slate-50 text-sm flex items-center gap-2 ${
                        s.name === current ? "bg-cream font-semibold" : ""
                      }`}
                      onClick={() => onSelect(s.name)}
                      {...hoverProps(s.name, s.reference_image_url)}
                    >
                      <SpeciesThumb url={s.reference_image_url} />
                      <span className="flex-1 min-w-0">{s.name}</span>
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

/**
 * Small reference thumbnail rendered inside each picker row. Falls back
 * to a slate placeholder square (with an optional emoji glyph) when the
 * species has no URL yet, so the row layout stays consistent. Width is
 * fixed so the species names align vertically across the list.
 */
function SpeciesThumb({
  url,
  fallback,
}: {
  url?: string | null;
  fallback?: string;
}) {
  if (url) {
    return (
      <img
        src={url}
        aria-hidden
        className="w-20 h-20 rounded object-cover border border-slate-200 flex-shrink-0"
        loading="lazy"
        onError={(e) => {
          // Hot-link blocked or URL stale — collapse to the placeholder.
          const img = e.currentTarget as HTMLImageElement;
          img.style.display = "none";
        }}
      />
    );
  }
  return (
    <div
      aria-hidden
      className="w-20 h-20 rounded border border-slate-200 bg-slate-50 flex-shrink-0 flex items-center justify-center text-slate-300 text-2xl"
    >
      {fallback ?? "🐦"}
    </div>
  );
}
