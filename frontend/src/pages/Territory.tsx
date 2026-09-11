import { useCallback, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { TerritoryMap, formatSeconds } from "../components/TerritoryMap";
import { ChevronIcon } from "../components/FieldIcons";
import {
  fetchTerritoryDates,
  fetchTerritoryDay,
  type TerritoryDay,
  type TerritoryZone,
} from "../lib/api";

/**
 * Territory: Feeder Wars. One day of footage read as a turf war.
 *
 * The page's job beyond drawing the map is to keep two claims apart, since
 * conflating them is the easiest way for this surface to start lying:
 *
 *   - Occupancy is cheap. Every detection has a bbox, so every bird can be
 *     placed in a zone.
 *   - A pecking order is expensive. Saying one bird pushed another off a
 *     perch needs both tracks on a shared clock, which needs
 *     `Detection.track_frames`. That column arrived with this page, so the
 *     archive cannot produce a single displacement and the footage from
 *     here on can. The Ledger block below says so in as many words rather
 *     than letting an empty log read as a peaceful yard.
 */
export default function Territory() {
  const [pickedDate, setPickedDate] = useState<string | null>(null);
  const [zone, setZone] = useState<TerritoryZone | null>(null);

  const datesQ = useQuery({
    queryKey: ["territory-dates"],
    queryFn: () => fetchTerritoryDates(60),
  });
  const dates = datesQ.data?.dates ?? [];
  const date = pickedDate ?? dates[0]?.date ?? null;

  const dayQ = useQuery({
    queryKey: ["territory-day", date],
    queryFn: () => fetchTerritoryDay({ date: date! }),
    enabled: date !== null,
  });

  const day = dayQ.data;

  const chooseDate = useCallback((next: string) => {
    setPickedDate(next);
    setZone(null);
  }, []);

  const dateIndex = dates.findIndex((d) => d.date === date);

  return (
    <div className="space-y-4 pb-8">
      <div>
        <div className="fg-overline">Dominance &amp; territory</div>
        <h2 className="font-serif font-medium text-2xl text-ink leading-tight mt-0.5">
          Feeder Wars
        </h2>
        <p className="text-sm text-muted mt-1">
          The camera frame, split into the four places birds actually land, and scored as ground
          held. Factions are drawn from what turns up in this yard rather than from a field guide.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="ml-auto flex items-center gap-1">
          <button
            className="fg-btn-ghost px-2 py-1.5 disabled:opacity-40"
            disabled={dateIndex < 0 || dateIndex >= dates.length - 1}
            onClick={() => chooseDate(dates[dateIndex + 1].date)}
            aria-label="Earlier day"
            title="Earlier day"
          >
            <ChevronIcon size={14} className="rotate-90" />
          </button>
          <select
            value={date ?? ""}
            onChange={(e) => chooseDate(e.target.value)}
            className="fg-input w-auto py-1.5 text-xs tnum"
          >
            {dates.map((d) => (
              <option key={d.date} value={d.date}>
                {d.date} — {d.detection_count} birds
                {d.timed_count > 0 ? `, ${d.timed_count} timed` : ""}
              </option>
            ))}
          </select>
          <button
            className="fg-btn-ghost px-2 py-1.5 disabled:opacity-40"
            disabled={dateIndex <= 0}
            onClick={() => chooseDate(dates[dateIndex - 1].date)}
            aria-label="Later day"
            title="Later day"
          >
            <ChevronIcon size={14} className="-rotate-90" />
          </button>
        </div>
      </div>

      <div className="relative">
        <TerritoryMap
          zones={day?.zones ?? []}
          basis={day?.control_basis ?? "appearances"}
          selected={zone?.id ?? null}
          onSelect={setZone}
        />
        {dayQ.isFetching && (
          <div className="absolute inset-0 grid place-items-center bg-surface/70 rounded-card">
            <span className="text-xs text-muted animate-pulse">Counting the ground…</span>
          </div>
        )}
      </div>

      {day && !day.summary.quiet && (
        <p className="fg-card p-3 font-serif text-[15px] text-ink leading-snug">{day.dispatch}</p>
      )}
      {day?.summary.quiet && (
        <p className="fg-card p-3 text-sm text-muted">
          The camera logged nothing on this day, so there was no ground to take.
        </p>
      )}

      {zone && <ZoneCard zone={zone} basis={day?.control_basis ?? "appearances"} onClear={() => setZone(null)} />}

      {day && !day.summary.quiet && (
        <>
          <Standings day={day} />
          <Log day={day} />
          <Ledger day={day} />
        </>
      )}

      {dayQ.error && (
        <p className="text-sm text-rust">The yard is not answering. The API may be down.</p>
      )}
    </div>
  );
}

// ── Pieces ──────────────────────────────────────────────────────────────

function ZoneCard({
  zone,
  basis,
  onClear,
}: {
  zone: TerritoryZone;
  basis: "seconds" | "appearances";
  onClear: () => void;
}) {
  return (
    <div className="fg-card p-3">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2 flex-wrap">
            <h3 className="font-serif text-lg text-ink leading-tight">{zone.name}</h3>
            {zone.contested && (
              <span className="text-[10px] uppercase tracking-wider text-rust">Contested</span>
            )}
          </div>
          <p className="text-sm text-muted mt-0.5">{zone.blurb}</p>
        </div>
        <button className="fg-btn-ghost px-2 py-1 text-xs shrink-0" onClick={onClear}>
          Clear
        </button>
      </div>

      {zone.control.length === 0 ? (
        <p className="text-sm text-faint mt-2">
          Nothing the classifier could name landed here, so nobody holds it.
        </p>
      ) : (
        <ul className="mt-2.5 space-y-1.5">
          {zone.control.map((c) => (
            <li key={c.faction} className="flex items-center gap-2 text-sm">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ background: c.color }}
                aria-hidden
              />
              <span className="text-ink min-w-0 flex-1 truncate">{c.name}</span>
              <span className="h-1.5 rounded-full bg-line w-24 shrink-0 overflow-hidden">
                <span
                  className="block h-full rounded-full"
                  style={{ width: `${Math.max(3, c.share * 100)}%`, background: c.color }}
                />
              </span>
              <span className="text-xs text-faint tnum shrink-0 w-20 text-right">
                {basis === "seconds"
                  ? formatSeconds(c.seconds)
                  : `${c.holds} landing${c.holds === 1 ? "" : "s"}`}
              </span>
            </li>
          ))}
        </ul>
      )}

      {zone.unclaimed > 0 && (
        <p className="text-xs text-faint mt-2">
          {zone.unclaimed} more {zone.unclaimed === 1 ? "bird" : "birds"} landed here that the
          classifier could not name. They hold ground for nobody.
        </p>
      )}
    </div>
  );
}

function Standings({ day }: { day: TerritoryDay }) {
  const top = day.standings[0]?.[day.control_basis === "seconds" ? "seconds" : "holds"] ?? 1;
  return (
    <div className="space-y-2">
      <div className="fg-overline">Standings</div>
      <ul className="space-y-1.5">
        {day.standings.map((s) => {
          const value = day.control_basis === "seconds" ? s.seconds : s.holds;
          return (
            <li key={s.faction} className="fg-card p-2.5">
              <div className="flex items-center gap-2.5">
                <span
                  className="w-2.5 h-2.5 rounded-full shrink-0"
                  style={{ background: s.color }}
                  aria-hidden
                />
                <span className="min-w-0 flex-1">
                  <span className="text-sm text-ink">{s.name}</span>
                  <span className="text-xs text-faint"> · {s.style}</span>
                </span>
                <span className="text-xs text-faint tnum shrink-0">
                  {s.zones_held > 0 &&
                    `${s.zones_held} zone${s.zones_held === 1 ? "" : "s"} · `}
                  {day.control_basis === "seconds"
                    ? formatSeconds(s.seconds)
                    : `${s.holds} landing${s.holds === 1 ? "" : "s"}`}
                </span>
              </div>
              <span className="mt-1.5 block h-1.5 rounded-full bg-line overflow-hidden">
                <span
                  className="block h-full rounded-full"
                  style={{
                    width: `${Math.max(2, (value / Math.max(top, 1)) * 100)}%`,
                    background: s.color,
                  }}
                />
              </span>
              {(s.wins > 0 || s.losses > 0) && (
                <div className="text-[11px] text-faint tnum mt-1">
                  {s.wins} perch{s.wins === 1 ? "" : "es"} taken, {s.losses} lost
                </div>
              )}
              {s.species.length > 0 && (
                <div className="text-[11px] text-faint mt-1 truncate">
                  {s.species.map((sp) => `${sp.species} (${sp.count})`).join(", ")}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Log({ day }: { day: TerritoryDay }) {
  return (
    <div className="space-y-2">
      <div className="fg-overline">Perches taken</div>
      {day.displacements.length === 0 ? (
        <p className="text-sm text-faint">
          {day.summary.eligible_visits === 0
            ? "No clip on this day had two timed birds in it, so nothing here could be judged either way."
            : "Birds shared the feeder on this day. Nobody was pushed off anything."}
        </p>
      ) : (
        <ul className="space-y-1.5">
          {day.displacements.map((e, i) => (
            <li key={`${e.visit_id}-${e.at_frame}-${i}`} className="fg-card p-2.5">
              <div className="text-sm text-ink">
                <span style={{ color: e.winner_color }}>{e.winner ?? "An unnamed bird"}</span> took{" "}
                {e.zone_name} from{" "}
                <span style={{ color: e.loser_color }}>{e.loser ?? "an unnamed bird"}</span>
              </div>
              <div className="text-[11px] text-faint tnum mt-0.5">
                {e.at_seconds.toFixed(1)}s into the clip, after {e.held_seconds.toFixed(1)}s held
                {e.winner_faction_name && ` · ${e.winner_faction_name}`}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** What the day's numbers rest on, and what they do not. */
function Ledger({ day }: { day: TerritoryDay }) {
  const s = day.summary;
  const untimed = s.detections - s.timed;
  return (
    <div className="fg-card p-3 text-sm text-muted space-y-1.5">
      <p>
        <span className="text-ink font-semibold tnum">{s.detections}</span> birds across{" "}
        <span className="text-ink font-semibold tnum">{s.visits}</span> clips,{" "}
        <span className="text-ink font-semibold tnum">{s.in_a_zone}</span> of them in one of the
        four zones.
      </p>
      <p className="text-xs text-faint">
        {day.control_basis === "seconds"
          ? "Ground is counted in seconds held, measured from each bird's per-frame track."
          : "No bird on this day carries a per-frame track, so ground is counted in landings rather than seconds held."}
        {day.zone_source === "built-in" &&
          " The zone map is the built-in one, read off the yard's own density heatmap. Move a feeder and it stops being true."}
      </p>
      <p className="text-xs text-faint">
        {s.eligible_visits === 0 ? (
          <>
            No perch could change hands on this day: judging that needs two birds in one clip with
            per-frame timings, and {untimed === s.detections ? "none" : `${untimed}`} of the day's
            birds carry one. Footage captured from now on does.
          </>
        ) : (
          <>
            <span className="text-ink tnum">{s.contested_visits}</span> clips had more than one
            bird in them, and <span className="text-ink tnum">{s.eligible_visits}</span> had two
            with timings good enough to say who moved whom.
          </>
        )}
      </p>
      {s.unclaimed > 0 && (
        <p className="text-xs text-faint">
          <span className="text-ink tnum">{s.unclaimed}</span> birds landed in a zone without a
          species the classifier would commit to. They are counted as traffic and never as
          somebody's ground.
        </p>
      )}
    </div>
  );
}
