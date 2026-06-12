import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";

import { fetchSpecies, type FamilyEntry, type SpeciesEntry } from "../lib/api";
import { BanIcon, BirdMark, FogIcon } from "./FieldIcons";

// Must match backend/db/models.py:NOT_A_BIRD_LABEL exactly.
export const NOT_A_BIRD = "Not a bird";
// Must match backend/db/models.py:UNKNOWN_BIRD_LABEL.
export const UNKNOWN_BIRD = "Unknown bird";
// Must match backend/db/models.py:POOR_QUALITY_LABEL. Stronger than
// UNKNOWN_BIRD — "this crop will never be identifiable" — and hidden
// from the default feed alongside NAB.
export const POOR_QUALITY = "Poor quality";

export type Suggestion = { species: string; p: number };

type Props = {
  open: boolean;
  current?: string | null;
  suggestions?: Suggestion[];
  cropUrl?: string;
  onSelect: (species: string) => void;
  onCancel: () => void;
};

/**
 * Searchable combobox over species labels. Renders families, yard species
 * (Haikubox-heard, with counts) and the broader NA list, plus a pinned
 * compare strip (user's crop vs. hovered species reference photo).
 */
export default function SpeciesPicker({
  open,
  current,
  suggestions,
  cropUrl,
  onSelect,
  onCancel,
}: Props) {
  // ≥ 0.5% so a suggestion never rounds down to a "0%" row — those are
  // pure noise on classifier-rejected (confidence 0) detections.
  const suggestionList = (suggestions ?? [])
    .filter((s) => s.species && s.species !== current && s.p >= 0.005)
    .slice(0, 3);
  const [query, setQuery] = useState("");
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

  const { familiesFiltered, yardFiltered, extraFiltered } = useMemo(() => {
    const empty = {
      familiesFiltered: [] as FamilyEntry[],
      yardFiltered: [] as SpeciesEntry[],
      extraFiltered: [] as SpeciesEntry[],
    };
    if (!data) return empty;
    const q = query.trim().toLowerCase();
    const matchSp = (s: SpeciesEntry) => !q || s.name.toLowerCase().includes(q);
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

  const firstMatch =
    familiesFiltered[0]?.name ?? yardFiltered[0]?.name ?? extraFiltered[0]?.name;

  const refUrlByName = useMemo(() => {
    const m = new Map<string, string | null>();
    if (!data) return m;
    for (const f of data.families ?? []) m.set(f.name, f.reference_image_url ?? null);
    for (const s of data.yard) m.set(s.name, s.reference_image_url ?? null);
    for (const s of data.extra) m.set(s.name, s.reference_image_url ?? null);
    return m;
  }, [data]);
  const currentRefUrl = current ? refUrlByName.get(current) ?? null : null;

  const hoverProps = (name: string, url: string | null | undefined) => ({
    onMouseEnter: () => setHovered({ name, url: url ?? null }),
    onFocus: () => setHovered({ name, url: url ?? null }),
  });

  const comparePane = hovered ?? (current ? { name: current, url: currentRefUrl } : null);

  // Keep the last reference URL we showed so the <img> elements in the
  // right compare pane stay mounted with a valid src across the
  // entire picker session. Without this, hovering from a row with a
  // photo to a sentinel (no photo) and back tears down the imgs and
  // reinserts them — the visible "flash + jitter" the user reports.
  const lastUrlRef = useRef<string | null>(null);
  if (comparePane?.url) lastUrlRef.current = comparePane.url;
  const stableUrl = comparePane?.url ?? lastUrlRef.current;

  if (!open) return null;

  const rowBase =
    "w-full text-left px-3.5 py-2.5 text-sm transition-colors hover:bg-[color-mix(in_oklab,var(--accent)_10%,transparent)]";
  const activeRow = "bg-[color-mix(in_oklab,var(--accent)_12%,transparent)] font-semibold text-ink";
  const sectionLabel = "fg-overline px-3.5 pt-3 pb-1 bg-panel/40 border-t border-line";

  // Render via portal so `position: fixed` is laid out against the
  // viewport, not against whichever ancestor happens to be transformed.
  // The parent DetectionCard uses `.fg-liftable`, which applies a
  // `translateY(-2px)` on hover — and `transform` creates a containing
  // block for fixed descendants, which trapped the picker inside the
  // card's bounds (tiny dialog) until the card unhovered (full dialog),
  // producing the flash + jitter as hover state toggled.
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-20 backdrop-blur-sm bg-[color-mix(in_oklab,var(--ink)_52%,transparent)]"
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="fg-card shadow-pop w-full max-w-lg mx-4 max-h-[80vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-3 border-b border-line">
          <input
            ref={inputRef}
            type="text"
            placeholder="Search species…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="fg-input"
            onKeyDown={(e) => {
              if (e.key === "Escape") onCancel();
              if (e.key === "Enter" && firstMatch) onSelect(firstMatch);
            }}
          />
        </div>

        {/* Compare strip: user's crop (left) vs. hovered/current reference. */}
        {cropUrl && (
          <div className="grid grid-cols-2 gap-px bg-line border-b border-line">
            <div className="relative bg-panel aspect-[4/3] overflow-hidden">
              <img
                src={cropUrl}
                alt="your crop"
                className="absolute inset-0 w-full h-full object-contain"
              />
              {/* z-[2] keeps the chip above the crop img; max-w + truncate
                  so it never paints past the pane at narrow widths. */}
              <span className="fg-overline absolute z-[2] top-1.5 left-1.5 max-w-[calc(100%-12px)] truncate px-1.5 py-0.5 rounded bg-surface/85">
                your crop
              </span>
            </div>
            <div className="relative bg-panel aspect-[4/3] overflow-hidden">
              {/* Stable DOM: the two <img>s and the label are always
                  mounted whenever we've ever shown a reference photo in
                  this picker session. `style.visibility` hides them when
                  the currently-hovered row has no URL, instead of
                  unmounting them — which is what was causing the flash
                  + jitter when hovering between rows-with-photos and
                  sentinels-without. The placeholder div is the only
                  conditionally-mounted element. */}
              {stableUrl && (
                <>
                  <img
                    src={stableUrl}
                    aria-hidden
                    className="absolute inset-0 w-full h-full object-cover blur-2xl scale-110 opacity-60"
                    style={{ visibility: comparePane?.url ? "visible" : "hidden" }}
                  />
                  <img
                    src={stableUrl}
                    alt={comparePane?.name ?? ""}
                    className="relative w-full h-full object-contain"
                    style={{ visibility: comparePane?.url ? "visible" : "hidden" }}
                  />
                </>
              )}
              {comparePane?.url && (
                <span className="absolute top-1.5 left-1.5 right-1.5 px-1.5 py-0.5 rounded bg-surface/85 text-[10px] font-semibold text-ink truncate">
                  {comparePane.name}
                </span>
              )}
              {!comparePane?.url && (
                <div className="absolute inset-0 flex items-center justify-center text-[11px] text-faint text-center px-3 whitespace-pre-line">
                  {comparePane
                    ? `${comparePane.name}\n(no reference photo)`
                    : "Hover a species to compare"}
                </div>
              )}
            </div>
          </div>
        )}

        <div className="overflow-y-auto flex-1" onMouseLeave={() => setHovered(null)}>
          {/* Pinned sentinel options. */}
          <button
            className={`${rowBase} flex items-center justify-between border-b border-line/60 text-muted`}
            onClick={() => onSelect(NOT_A_BIRD)}
            {...hoverProps(NOT_A_BIRD, null)}
          >
            <span className="flex items-center gap-2 text-rust">
              <BanIcon size={15} />
              <span>{NOT_A_BIRD}</span>
            </span>
            <span className="text-xs text-faint">false positive</span>
          </button>
          <button
            className={`${rowBase} flex items-center justify-between border-b border-line/60 text-muted`}
            onClick={() => onSelect(UNKNOWN_BIRD)}
            {...hoverProps(UNKNOWN_BIRD, null)}
          >
            <span className="flex items-center gap-2">
              <span
                className="grid place-items-center w-[15px] h-[15px] rounded-full border border-faint text-[10px] font-bold text-faint"
                aria-hidden
              >
                ?
              </span>
              <span>{UNKNOWN_BIRD}</span>
            </span>
            <span className="text-xs text-faint">bird, species unsure</span>
          </button>
          <button
            className={`${rowBase} flex items-center justify-between border-b border-line text-muted`}
            onClick={() => onSelect(POOR_QUALITY)}
            {...hoverProps(POOR_QUALITY, null)}
          >
            <span className="flex items-center gap-2">
              <FogIcon size={15} />
              <span>{POOR_QUALITY}</span>
            </span>
            <span className="text-xs text-faint">unidentifiable crop</span>
          </button>

          {/* Classifier's next-best guesses. */}
          {suggestionList.length > 0 && !query && (
            <>
              <div className={sectionLabel}>Suggested by classifier</div>
              <ul>
                {suggestionList.map((s) => (
                  <li key={`suggestion-${s.species}`}>
                    <button
                      className={`${rowBase} flex items-center justify-between border-b border-line/40`}
                      onClick={() => onSelect(s.species)}
                      {...hoverProps(s.species, refUrlByName.get(s.species) ?? null)}
                    >
                      <span className="flex items-center gap-2">
                        <i
                          className="w-1.5 h-1.5 rounded-full"
                          style={{ background: "var(--accent)" }}
                          aria-hidden
                        />
                        <span>{s.species}</span>
                      </span>
                      <span className="text-xs text-faint tnum">{Math.round(s.p * 100)}%</span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}

          {isLoading && <p className="p-3 text-sm text-muted">Loading species…</p>}
          {!isLoading &&
            familiesFiltered.length === 0 &&
            yardFiltered.length === 0 &&
            extraFiltered.length === 0 && (
              <p className="p-3 text-sm text-muted">No species match "{query}".</p>
            )}

          {/* Family-level catch-alls. */}
          {familiesFiltered.length > 0 && (
            <>
              <div className={sectionLabel}>Families</div>
              <ul>
                {familiesFiltered.map((f) => (
                  <li key={`family-${f.name}`}>
                    <button
                      className={`${rowBase} flex items-start gap-2.5 ${
                        f.name === current ? activeRow : ""
                      }`}
                      onClick={() => onSelect(f.name)}
                      {...hoverProps(f.name, f.reference_image_url)}
                    >
                      <SpeciesThumb url={f.reference_image_url} group />
                      <span className="flex-1 min-w-0 pt-0.5">
                        <span className="block">{f.name}</span>
                        <span className="block text-xs text-faint truncate">
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

          {/* Yard species. */}
          {yardFiltered.length > 0 && (
            <>
              <div className="fg-overline px-3.5 pt-3 pb-1 bg-panel/40">Heard in this yard</div>
              <ul>
                {yardFiltered.map((s) => (
                  <li key={`yard-${s.name}`}>
                    <button
                      className={`${rowBase} flex items-center gap-2.5 ${
                        s.name === current ? activeRow : ""
                      }`}
                      onClick={() => onSelect(s.name)}
                      {...hoverProps(s.name, s.reference_image_url)}
                    >
                      <SpeciesThumb url={s.reference_image_url} />
                      <span className="flex-1 min-w-0">{s.name}</span>
                      {s.total > 0 && (
                        <span className="text-xs text-faint flex-shrink-0 tnum">
                          {s.total >= 1000 ? `${Math.round(s.total / 1000)}k` : s.total}
                        </span>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}

          {/* Broader NA list. */}
          {extraFiltered.length > 0 && (
            <>
              <div className={sectionLabel}>Other North American species</div>
              <ul>
                {extraFiltered.map((s) => (
                  <li key={`extra-${s.name}`}>
                    <button
                      className={`${rowBase} flex items-center gap-2.5 ${
                        s.name === current ? activeRow : ""
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

        <div className="p-2.5 border-t border-line flex justify-between items-center text-xs text-muted">
          <span>
            {data?.source === "calibration"
              ? `${data.yard.length} from yard · ${data.extra.length} other NA species`
              : "Fallback species list (calibrate for better picks)."}
          </span>
          <button onClick={onCancel} className="text-muted hover:text-ink transition-colors">
            Cancel
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

/**
 * Reference thumbnail inside each picker row. Falls back to a panel
 * placeholder with a faint field-guide bird mark (or a "group" glyph for
 * family rows) so row layout stays consistent.
 */
function SpeciesThumb({ url, group }: { url?: string | null; group?: boolean }) {
  if (url) {
    return (
      <img
        src={url}
        aria-hidden
        className="w-16 h-16 rounded-md object-cover border border-line flex-shrink-0"
        loading="lazy"
        onError={(e) => {
          (e.currentTarget as HTMLImageElement).style.display = "none";
        }}
      />
    );
  }
  return (
    <div
      aria-hidden
      className="w-16 h-16 rounded-md border border-line bg-panel flex-shrink-0 grid place-items-center text-faint"
    >
      {group ? (
        <span className="text-xs font-semibold">grp</span>
      ) : (
        <BirdMark size={22} />
      )}
    </div>
  );
}
