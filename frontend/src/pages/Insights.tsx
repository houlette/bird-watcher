import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fetchSpeciesActivity, type SpeciesActivity } from "../lib/api";
import { readTokens, tip } from "../lib/chartTheme";
import { ChevronIcon } from "../components/FieldIcons";

// ─── Axis label helpers ─────────────────────────────────────────────────────
// Hour ticks every 3 hours, in friendly 12-hour form ("6a", "12p"). Other
// hours render blank so the 24-bar axis stays legible.
function hourTick(h: number): string {
  if (h % 3 !== 0) return "";
  const ampm = h < 12 ? "a" : "p";
  const twelve = h % 12 === 0 ? 12 : h % 12;
  return `${twelve}${ampm}`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// Week bucket i covers days [i*7, i*7+7) from Jan 1 (see backend
// compute_species_activity). Label a bucket with the month abbreviation only
// when it's the first bucket whose start date falls in a new month — gives a
// clean Jan…Dec axis under the 53 bars. Reference year is arbitrary
// (non-leap) since we only need month boundaries.
const WEEK_TICKS: string[] = (() => {
  const out: string[] = [];
  let prevMonth = -1;
  for (let i = 0; i < 53; i++) {
    const d = new Date(Date.UTC(2025, 0, 1 + i * 7));
    const m = d.getUTCMonth();
    out.push(m !== prevMonth ? MONTHS[m] : "");
    prevMonth = m;
  }
  return out;
})();

// ─── Layout primitives (mirrors Stats.tsx) ──────────────────────────────────
function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`fg-card p-4 ${className}`}>{children}</section>;
}
function CardTitle({ children, hint }: { children: React.ReactNode; hint?: React.ReactNode }) {
  return (
    <h3 className="font-serif text-[17px] font-medium text-ink mb-1">
      {children}
      {hint && <span className="ml-1.5 text-xs text-faint font-sans font-normal">{hint}</span>}
    </h3>
  );
}

// ─── Species picker ──────────────────────────────────────────────────────────
// Self-contained modal that lists the species in the activity payload (already
// sorted by total). No extra fetch — the picker and the charts share one query.
function SpeciesPickerButton({
  species,
  selectedId,
  onSelect,
}: {
  species: SpeciesActivity[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Focusing the input is a legit effect (touching a DOM API); the query
  // reset happens in the open handler so we don't setState inside the effect.
  useEffect(() => {
    if (open) requestAnimationFrame(() => inputRef.current?.focus());
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? species.filter((s) => s.species.toLowerCase().includes(q)) : species;
  }, [species, query]);

  const selected = species.find((s) => s.species_id === selectedId);
  const rowBase =
    "w-full text-left px-3.5 py-2.5 text-sm border-b border-line/60 transition-colors hover:bg-[color-mix(in_oklab,var(--accent)_10%,transparent)]";
  const activeRow = "bg-[color-mix(in_oklab,var(--accent)_12%,transparent)] font-semibold text-ink";

  return (
    <>
      <button
        type="button"
        className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1.5 text-sm text-muted hover:border-leaf hover:text-leaf transition-colors"
        onClick={() => {
          setQuery("");
          setOpen(true);
        }}
        title="Choose a species"
      >
        <span className="text-faint">Species:</span>
        <span className="font-semibold text-ink">{selected ? selected.species : "Select…"}</span>
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
              <div className="overflow-y-auto flex-1">
                {filtered.length === 0 && (
                  <p className="p-3 text-sm text-muted">No species match "{query}".</p>
                )}
                <ul>
                  {filtered.map((s) => (
                    <li key={s.species_id}>
                      <button
                        className={`${rowBase} flex items-center justify-between gap-2 ${
                          s.species_id === selectedId ? activeRow : ""
                        }`}
                        onClick={() => {
                          onSelect(s.species_id);
                          setOpen(false);
                        }}
                      >
                        <span>{s.species}</span>
                        <span className="text-xs text-faint tnum" title="sightings">
                          {s.total}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}

// ─── Charts ──────────────────────────────────────────────────────────────────
function TimeOfDayChart({ sp }: { sp: SpeciesActivity }) {
  const t = readTokens();
  const rows = useMemo(
    () => sp.by_hour.map((count, hour) => ({ hour, label: hourTick(hour), count })),
    [sp],
  );
  return (
    <Card>
      <CardTitle hint="(feeder local time)">Time of day</CardTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={rows} margin={{ left: -12, right: 10, top: 8, bottom: 4 }}>
          <CartesianGrid stroke={t.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="hour"
            type="category"
            scale="band"
            tick={{ fontSize: 11, fill: t.axis }}
            interval={0}
            tickFormatter={(h) => hourTick(h as number)}
          />
          <YAxis tick={{ fontSize: 11, fill: t.axis }} allowDecimals={false} />
          <Tooltip
            {...tip(t)}
            labelFormatter={(h) => `${String(h).padStart(2, "0")}:00`}
            formatter={(v) => [`${v}`, "sightings"]}
          />
          {/* isAnimationActive={false}: recharts otherwise leaves bar
              heights stale when the data swaps on a species change (the
              y-axis rescales but the <path>s keep the prior values). */}
          <Bar dataKey="count" fill={t.leaf} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-muted mt-2">
        Sightings by hour, summed across all days. Taller bars are the hours this species is most
        active at the feeder.
      </p>
    </Card>
  );
}

function TimeOfYearChart({ sp }: { sp: SpeciesActivity }) {
  const t = readTokens();
  const rows = useMemo(
    () => sp.by_week.map((count, week) => ({ week, label: WEEK_TICKS[week], count })),
    [sp],
  );
  return (
    <Card>
      <CardTitle hint="(by week)">Time of year</CardTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={rows} margin={{ left: -12, right: 10, top: 8, bottom: 4 }}>
          <CartesianGrid stroke={t.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="week"
            type="category"
            scale="band"
            tick={{ fontSize: 11, fill: t.axis }}
            interval={0}
            tickFormatter={(w) => WEEK_TICKS[w as number] ?? ""}
          />
          <YAxis tick={{ fontSize: 11, fill: t.axis }} allowDecimals={false} />
          <Tooltip
            {...tip(t)}
            labelFormatter={(w) => {
              const d = new Date(Date.UTC(2025, 0, 1 + (w as number) * 7));
              return `Week of ${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
            }}
            formatter={(v) => [`${v}`, "sightings"]}
          />
          <Bar dataKey="count" fill={t.sand} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-muted mt-2">
        Sightings by week of the year. This view fills in as the seasons pass — arrivals and
        departures (e.g. a spring-only oriole) show up here.
      </p>
    </Card>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────
export default function Insights() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["species-activity"],
    queryFn: fetchSpeciesActivity,
    refetchInterval: 5 * 60 * 1000,
  });

  // User's explicit pick, or null to mean "use the default (most-seen)".
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const species = data?.species ?? [];

  if (isLoading) return <p className="text-muted mt-4">Loading activity…</p>;
  if (error || !data) {
    return (
      <p className="text-rust text-sm mt-4">
        Couldn't load activity: {(error as Error)?.message ?? "unknown error"}
      </p>
    );
  }
  if (species.length === 0) {
    return (
      <div className="mt-4">
        <p className="font-serif italic text-lg text-muted">No species sightings yet.</p>
        <p className="text-sm text-faint mt-1">
          Activity charts appear once the feeder logs identified birds.
        </p>
      </div>
    );
  }

  const selected = species.find((s) => s.species_id === selectedId) ?? species[0];
  const fmtDate = (iso: string) =>
    new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });

  return (
    <div className="space-y-4">
      <div className="mb-1">
        <div className="fg-overline">Visit patterns</div>
        <h2 className="font-serif font-medium text-2xl text-ink leading-tight mt-0.5">
          When species visit
        </h2>
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <SpeciesPickerButton
          species={species}
          selectedId={selected.species_id}
          onSelect={setSelectedId}
        />
        <span className="text-sm text-faint tnum">
          {selected.total} sighting{selected.total === 1 ? "" : "s"}
        </span>
      </div>

      <Card>
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <div>
            <h3 className="font-serif text-2xl text-ink leading-tight">{selected.species}</h3>
            {selected.scientific_name && (
              <div className="font-serif italic text-muted mt-0.5">{selected.scientific_name}</div>
            )}
          </div>
          <div className="text-xs text-faint text-right">
            <div>
              First seen <span className="text-muted tnum">{fmtDate(selected.first_seen)}</span>
            </div>
            <div>
              Last seen <span className="text-muted tnum">{fmtDate(selected.last_seen)}</span>
            </div>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <TimeOfDayChart sp={selected} />
        <TimeOfYearChart sp={selected} />
      </div>

      <p className="text-xs text-faint">
        Each sighting is one identified bird (a visit with several of the same species counts each
        one). Times are binned in the feeder's local zone ({data.tz}). Labels reflect your
        corrections where you've made them.
      </p>
      <p className="text-xs text-faint text-right tnum">
        Updated {new Date(data.as_of).toLocaleString()}
      </p>
    </div>
  );
}
