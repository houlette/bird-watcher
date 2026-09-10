import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  TavernCanvas,
  type TavernCanvasHandle,
  type Phase,
} from "../components/TavernCanvas";
import { SoundIcon, VolumeIcon, VolumeMuteIcon } from "../components/FieldIcons";
import {
  buyTavernUpgrade,
  fetchTavernArrivals,
  fetchTavernGuestbook,
  fetchTavernState,
  markTavernSeen,
  type TavernEvent,
  type TavernGuestbookEntry,
  type TavernPatron,
  type TavernUpgrade,
} from "../lib/api";
import { useThemeKey } from "../lib/useThemeKey";

/**
 * The Perch & Flagon: the feeder as a tavern, and its visitors as guests.
 *
 * The page holds the room's roster. A first load asks /state for whoever
 * arrived recently, then a poll asks /arrivals for anything new and seats
 * it. The canvas animates whatever roster it is handed and knows nothing
 * about fetching.
 *
 * The balance shown is the server's, plus whatever the poll has taken in
 * since the last full state read. That local addition is a display
 * convenience only: the next /state recomputes the whole ledger from the
 * detections table and overwrites it.
 */

/** How many guests the common room seats before the earliest one leaves. */
const ROOM_CAPACITY = 12;
const ARRIVAL_POLL_MS = 20_000;
const STATE_POLL_MS = 60_000;
const LOG_LENGTH = 40;

/**
 * Wall-clock seconds per beat of a patron's dwell.
 *
 * The server sends a `dwell` in beats, from the archetype and nudged by
 * how long the bird really stayed: about 0.75 for a messenger who barely
 * stopped, about 5 for a dove that settled. At ninety seconds a beat that
 * is roughly a minute for the chickadee and seven for the dove, which is
 * the spread the design asks for. The figure is a guess and this constant
 * is the one place to change it.
 */
const SECONDS_PER_BEAT = 90;

/**
 * Guests who never time out, counted from the newest backwards.
 *
 * Dwell alone empties the room on a yard that logs a few dozen birds a
 * day, and an empty tavern is the thing this page exists not to be. The
 * newest few keep their seats however long they have been there, so the
 * fire always has company and the churn happens behind them.
 */
const MIN_COMPANY = 3;

type LogItem =
  | { kind: "patron"; id: number; at: string; patron: TavernPatron }
  | { kind: "event"; id: number; at: string; event: TavernEvent };

type Tab = "room" | "upgrades" | "guestbook";

export default function Tavern() {
  const themeKey = useThemeKey();
  const queryClient = useQueryClient();
  const canvasRef = useRef<TavernCanvasHandle | null>(null);

  const [strangers, setStrangers] = useState(true);
  const [dock, setDock] = useState(false);
  const [sound, setSound] = useState(false);
  const [tab, setTab] = useState<Tab>("room");
  const [picked, setPicked] = useState<TavernPatron | null>(null);

  const [roster, setRoster] = useState<TavernPatron[]>([]);
  const [log, setLog] = useState<LogItem[]>([]);
  const [cursor, setCursor] = useState<number | null>(null);
  // Shillings taken since the last full ledger read, held locally so the
  // balance moves the moment a guest sits down.
  const [takings, setTakings] = useState(0);
  const [buying, setBuying] = useState<string | null>(null);
  const [shopError, setShopError] = useState<string | null>(null);

  const chime = useTavernSound(sound);

  /**
   * When the page sat each guest down, by detection id.
   *
   * Dwell is counted from this rather than from the bird's real arrival
   * time, because the room is a re-enactment and not a rewind: a dove the
   * camera saw an hour ago should walk in, sit, and leave, not appear
   * already overdue. An id that has been seated once stays in this map
   * after the guest has gone, which is what stops a /state refetch from
   * walking the same bird back through the door every minute.
   */
  const seatedAtRef = useRef<Map<number, number>>(new Map());

  /**
   * Seat guests without disturbing the ones already in the room. The canvas
   * keys its actors on detection id, so merging by id leaves everybody
   * where they are sitting and only the newcomers walk in.
   */
  const seat = useCallback((patrons: TavernPatron[]) => {
    const seatedAt = seatedAtRef.current;
    const now = Date.now();

    // Only guests nobody has seated yet. Anyone already in the map either
    // is in the room or has had their turn and left.
    const arriving = patrons.filter((p) => !seatedAt.has(p.detection_id));
    if (!arriving.length) return;
    for (const p of arriving) seatedAt.set(p.detection_id, now);

    // The map is the record of everyone ever seated, so trim the oldest
    // half when it gets long. A day of arrivals will not reach this.
    if (seatedAt.size > 600) {
      const oldest = [...seatedAt.entries()].sort((a, b) => a[1] - b[1]);
      for (const [id] of oldest.slice(0, 300)) seatedAt.delete(id);
    }

    setRoster((prev) => {
      // A live arrival ends the vigil: the last company the camera saw
      // gives up its seats to a bird that is actually here.
      const live = arriving.some((p) => !p.stale);
      const base = live ? prev.filter((p) => !p.stale) : prev;

      const merged = new Map(base.map((p) => [p.detection_id, p]));
      for (const p of arriving) merged.set(p.detection_id, p);
      return [...merged.values()]
        .sort((a, b) => a.arrived_at.localeCompare(b.arrived_at))
        .slice(-ROOM_CAPACITY);
    });
  }, []);

  // Show the door to anyone whose dwell has run out. The canvas watches
  // the roster, so dropping a guest here is what makes them stand up and
  // walk out. Returning the previous array unchanged when nobody is due
  // keeps this from re-rendering the page every few seconds.
  useEffect(() => {
    const timer = setInterval(() => {
      setRoster((prev) => {
        const next = prev.filter((p, i) => {
          if (p.stale) return true; // the quiet company keeps its seats
          if (i >= prev.length - MIN_COMPANY) return true;
          const seated = seatedAtRef.current.get(p.detection_id);
          if (seated === undefined) return true;
          return Date.now() - seated < p.dwell * SECONDS_PER_BEAT * 1000;
        });
        return next.length === prev.length ? prev : next;
      });
    }, 3000);
    return () => clearInterval(timer);
  }, []);

  // The room is updated where the data lands, in the fetch itself, rather
  // than in an effect watching the query result. Both run after the
  // request resolves, but this way there is one obvious place where a
  // guest is seated, and no render pass whose only job is to notice that
  // the last one produced new data.
  const stateQ = useQuery({
    queryKey: ["tavern-state", strangers],
    queryFn: async () => {
      const data = await fetchTavernState({
        strangers,
        limit: ROOM_CAPACITY,
        window_minutes: 180,
      });
      seat(data.patrons);
      setLog((prev) => (prev.length ? prev : seedLog(data.patrons, data.events)));
      setCursor((c) => c ?? data.cursor);
      // This ledger already counts everything the poll has added locally.
      setTakings(0);
      return data;
    },
    refetchInterval: STATE_POLL_MS,
    // The strangers toggle is part of the key, so without this the purse
    // and the shop empty out for as long as the refetch takes and the
    // house reads as broke.
    placeholderData: keepPreviousData,
  });

  // Run for its effects: the poll is what seats new guests.
  useQuery({
    queryKey: ["tavern-arrivals", cursor, strangers],
    queryFn: async () => {
      const from = cursor!;
      const data = await fetchTavernArrivals(from, strangers);

      if (data.patrons.length) {
        seat(data.patrons);
        setTakings((t) => t + data.earned);
        for (const p of data.patrons) chime(p);
        void markTavernSeen(data.patrons[data.patrons.length - 1].detection_id);
      }
      if (data.events.length) canvasRef.current?.gutter();
      if (data.patrons.length || data.events.length) {
        setLog((prev) => [...seedLog(data.patrons, data.events), ...prev].slice(0, LOG_LENGTH));
      }
      if (data.cursor > from) setCursor(data.cursor);
      return data;
    },
    enabled: cursor !== null,
    refetchInterval: ARRIVAL_POLL_MS,
  });

  const guestbookQ = useQuery({
    queryKey: ["tavern-guestbook"],
    queryFn: fetchTavernGuestbook,
    enabled: tab === "guestbook",
  });

  // The toggle filters the room as well as the fetch. Without this it only
  // governed who arrived next, so pressing "Hide strangers" left every
  // stranger already at the bar exactly where they were.
  const company = useMemo(
    () => (strangers ? roster : roster.filter((p) => p.species !== null)),
    [roster, strangers]
  );

  const ledger = stateQ.data?.ledger;
  const balance = (ledger?.balance ?? 0) + takings;
  const phase: Phase = stateQ.data?.hearth.phase ?? "day";
  const unlocked = stateQ.data?.unlocked ?? [];

  const upgrades = useMemo(() => {
    const list = stateQ.data?.upgrades ?? [];
    // Re-derive affordability against the balance the page is showing, so
    // a guest who just walked in enables the button they just paid for.
    return list.map((u) => ({ ...u, affordable: !u.owned && balance >= u.cost }));
  }, [stateQ.data, balance]);

  const buy = useCallback(
    async (upgrade: TavernUpgrade) => {
      setBuying(upgrade.id);
      setShopError(null);
      try {
        await buyTavernUpgrade(upgrade.id);
        await queryClient.invalidateQueries({ queryKey: ["tavern-state"] });
        await queryClient.invalidateQueries({ queryKey: ["tavern-guestbook"] });
      } catch (err) {
        setShopError(err instanceof Error ? err.message : "The purchase did not go through");
      } finally {
        setBuying(null);
      }
    },
    [queryClient]
  );

  const inRoom = company.filter((p) => !p.stale).length;

  return (
    <div className="space-y-4 pb-8">
      <div>
        <div className="fg-overline">Idle companion</div>
        <h2 className="font-serif font-medium text-2xl text-ink leading-tight mt-0.5">
          The Perch &amp; Flagon
        </h2>
        <p className="text-sm text-muted mt-1">
          The feeder, run as a tavern. Every detection the camera makes walks through the door as
          a guest, orders something, and leaves a few shillings on the table.
        </p>
      </div>

      {/* ── The room ───────────────────────────────────────────────────── */}
      <div className="relative">
        <TavernCanvas
          ref={canvasRef}
          patrons={company}
          unlocked={unlocked}
          phase={phase}
          themeKey={themeKey}
          pickedId={picked?.detection_id ?? null}
          onPick={setPicked}
          compact={dock}
        />
        {stateQ.isLoading && (
          <div className="absolute inset-0 grid place-items-center bg-surface/70 rounded-card">
            <span className="text-xs text-muted animate-pulse">Lighting the fire…</span>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {/* Null until the first ledger lands: a real zero and an unknown
            balance should not look the same. */}
        <Purse balance={ledger ? balance : null} />

        <span className="text-xs text-muted">
          {stateQ.data?.quiet
            ? "The room is quiet. This was the last company the camera saw."
            : `${inRoom} ${inRoom === 1 ? "guest" : "guests"} in the common room`}
        </span>

        <div className="ml-auto flex items-center gap-1.5">
          <button
            onClick={() => setStrangers((v) => !v)}
            className={`fg-btn-ghost px-2.5 py-1.5 text-xs ${
              strangers ? "" : "text-leaf border-leaf"
            }`}
            title={
              strangers
                ? "Hide the crops the classifier would not name"
                : "Seat the unnamed crops again"
            }
          >
            {strangers ? "Hide strangers" : "Strangers hidden"}
          </button>
          <button
            onClick={() => setSound((v) => !v)}
            className={`fg-btn-ghost px-2 py-1.5 ${sound ? "text-leaf border-leaf" : ""}`}
            title={sound ? "Silence the room" : "Play a note as each guest arrives"}
            aria-label={sound ? "Mute the tavern" : "Unmute the tavern"}
          >
            {sound ? <VolumeIcon size={14} /> : <VolumeMuteIcon size={14} />}
          </button>
          <button
            onClick={() => setDock((v) => !v)}
            className={`fg-btn-ghost px-2.5 py-1.5 text-xs ${dock ? "text-leaf border-leaf" : ""}`}
            title="A short room that sits at the bottom of a screen while you work"
          >
            {dock ? "Full room" : "Dock"}
          </button>
        </div>
      </div>

      {picked && <PatronCard patron={picked} onClear={() => setPicked(null)} />}

      {/* Dock mode is meant to sit in the corner of a screen, so it stops
          at the room, the purse and whoever you tapped on. */}
      {!dock && (
        <>
          <div className="flex items-center gap-1 border-b border-line">
            {(
              [
                ["room", "Common room"],
                ["upgrades", "The house"],
                ["guestbook", "Guestbook"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={`px-3 py-2 text-xs font-semibold tracking-wide border-b-2 -mb-px transition-colors ${
                  tab === id
                    ? "border-leaf text-leaf"
                    : "border-transparent text-faint hover:text-muted"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "room" && (
            <RoomLog
              log={log}
              today={stateQ.data?.today}
              ledger={ledger}
              sinceLastSeen={stateQ.data?.since_last_seen ?? 0}
            />
          )}
          {tab === "upgrades" && (
            <Shop
              upgrades={upgrades}
              balance={balance}
              buying={buying}
              error={shopError}
              onBuy={buy}
            />
          )}
          {tab === "guestbook" && (
            <Guestbook
              entries={guestbookQ.data?.entries ?? []}
              hasLedger={!!guestbookQ.data?.has_ledger}
              loading={guestbookQ.isLoading}
            />
          )}
        </>
      )}

      {stateQ.error && (
        <p className="text-sm text-rust">The tavern is not answering. The API may be down.</p>
      )}
    </div>
  );
}

// ── Pieces ──────────────────────────────────────────────────────────────

function Purse({ balance }: { balance: number | null }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-line bg-surface text-sm"
      title="Seed Shillings: what the birds have paid, less what the house has spent"
    >
      <span className="w-3 h-3 rounded-full bg-[#c9a227] border border-[#8a6f18]" aria-hidden />
      <span className="tnum font-semibold text-ink">
        {balance === null ? "—" : balance.toLocaleString()}
      </span>
      <span className="text-xs text-muted">shillings</span>
    </span>
  );
}

function PatronCard({ patron, onClear }: { patron: TavernPatron; onClear: () => void }) {
  return (
    <div className="fg-card p-3 flex gap-3">
      {patron.crop_url ? (
        <img
          src={patron.crop_url}
          alt={patron.species ?? "Unnamed visitor"}
          className="w-16 h-16 rounded-lg object-cover shrink-0"
        />
      ) : (
        <div
          className="w-16 h-16 rounded-lg shrink-0"
          style={{ background: patron.style.primary }}
        />
      )}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap">
          <h3 className="font-serif text-lg text-ink leading-tight">
            {patron.species ?? "A cloaked stranger"}
          </h3>
          {patron.species && (
            <span className="text-[10px] uppercase tracking-wider text-faint">
              {patron.archetype.title}
            </span>
          )}
          {patron.audio_confirmed && (
            <span className="inline-flex items-center gap-1 text-[10px] text-leaf uppercase tracking-wider">
              <SoundIcon size={11} /> Heard too
            </span>
          )}
        </div>
        <p className="font-serif italic text-muted text-sm mt-0.5">“{patron.line}”</p>
        <div className="text-xs text-faint mt-1">
          {patron.order.drink} and {patron.order.dish}
        </div>
        <div className="text-xs text-faint tnum mt-1">
          {formatTime(patron.arrived_at)}
          {" · "}
          {patron.rarity} here
          {" · "}
          paid {patron.payment}
          {patron.stale && " · not in the room now"}
        </div>
      </div>
      <button className="fg-btn-ghost px-2 py-1 text-xs self-start" onClick={onClear}>
        Clear
      </button>
    </div>
  );
}

function RoomLog({
  log,
  today,
  ledger,
  sinceLastSeen,
}: {
  log: LogItem[];
  today?: { arrivals: number; species: number; top: { species: string; count: number }[] };
  ledger?: { founding_purse: number; purse_cap: number; since_opening: number; patrons_served: number };
  sinceLastSeen: number;
}) {
  return (
    <div className="space-y-3">
      {today && (
        <div className="fg-card p-3 text-sm text-muted">
          <span className="text-ink font-semibold tnum">{today.arrivals}</span> through the door
          today, across <span className="text-ink font-semibold tnum">{today.species}</span>{" "}
          {today.species === 1 ? "species" : "species"}
          {today.top.length > 0 && (
            <>
              {". Busiest: "}
              {today.top.map((t) => `${t.species} (${t.count})`).join(", ")}
            </>
          )}
          .
          {sinceLastSeen > 0 && (
            <>
              {" "}
              <span className="text-ink font-semibold tnum">{sinceLastSeen.toLocaleString()}</span>{" "}
              came in since you last watched the door.
            </>
          )}
        </div>
      )}

      {log.length === 0 && (
        <p className="text-sm text-faint">Nobody has come in yet. The fire is lit anyway.</p>
      )}

      <ul className="space-y-2">
        {log.map((item) =>
          item.kind === "patron" ? (
            <li key={`p${item.id}`} className="fg-card p-2.5 flex items-start gap-2.5">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0 mt-1.5"
                style={{ background: item.patron.style.primary }}
                aria-hidden
              />
              <div className="min-w-0 flex-1">
                <div className="text-sm text-ink">
                  {item.patron.species ?? "A cloaked stranger"}
                  {/* A stranger's name IS their archetype; printing both
                      gives "A cloaked stranger · Cloaked stranger". */}
                  {item.patron.species && (
                    <span className="text-faint text-xs"> · {item.patron.archetype.title}</span>
                  )}
                </div>
                <p className="font-serif italic text-muted text-sm">“{item.patron.line}”</p>
              </div>
              <div className="text-right shrink-0">
                <div className="text-xs text-faint tnum">{formatTime(item.at)}</div>
                <div className="text-xs text-leaf tnum">+{item.patron.payment}</div>
              </div>
            </li>
          ) : (
            <li key={`e${item.id}`} className="fg-card p-2.5 border-dashed">
              <div className="flex items-baseline gap-2">
                <span className="text-sm text-rust font-semibold">{item.event.title}</span>
                <span className="ml-auto text-xs text-faint tnum">{formatTime(item.at)}</span>
              </div>
              <p className="text-sm text-muted">{item.event.line}</p>
              <p className="text-[11px] text-faint mt-1">
                The camera caught something the review marked as not a bird.
              </p>
            </li>
          )
        )}
      </ul>

      {ledger && (
        <p className="text-xs text-faint">
          The house opened with a founding purse of {ledger.founding_purse.toLocaleString()}{" "}
          shillings, the most the strongbox holds however long the archive behind it runs. It has
          taken {ledger.since_opening.toLocaleString()} more since opening, and served{" "}
          {ledger.patrons_served.toLocaleString()} guests in all.
        </p>
      )}
    </div>
  );
}

function Shop({
  upgrades,
  balance,
  buying,
  error,
  onBuy,
}: {
  upgrades: TavernUpgrade[];
  balance: number;
  buying: string | null;
  error: string | null;
  onBuy: (u: TavernUpgrade) => void;
}) {
  const categories = useMemo(() => {
    const groups = new Map<string, TavernUpgrade[]>();
    for (const u of upgrades) {
      const list = groups.get(u.category) ?? [];
      list.push(u);
      groups.set(u.category, list);
    }
    return [...groups.entries()];
  }, [upgrades]);

  return (
    <div className="space-y-4">
      {error && <p className="text-sm text-rust">{error}</p>}
      {categories.map(([category, items]) => (
        <div key={category} className="space-y-2">
          <div className="fg-overline">{category}</div>
          {items.map((u) => (
            <div key={u.id} className="fg-card p-3 flex items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2 flex-wrap">
                  <h3 className="font-serif text-base text-ink">{u.name}</h3>
                  {u.owned && (
                    <span className="text-[10px] uppercase tracking-wider text-leaf">
                      Part of the house
                    </span>
                  )}
                </div>
                <p className="text-sm text-muted mt-0.5">{u.blurb}</p>
              </div>
              <div className="shrink-0 text-right">
                <div className="tnum text-sm text-ink">{u.cost.toLocaleString()}</div>
                {!u.owned && (
                  <button
                    onClick={() => onBuy(u)}
                    disabled={!u.affordable || buying === u.id}
                    className="fg-btn-primary px-2.5 py-1 text-xs mt-1 disabled:opacity-40"
                    title={
                      u.affordable
                        ? `Spend ${u.cost} shillings`
                        : `${(u.cost - balance).toLocaleString()} more shillings needed`
                    }
                  >
                    {buying === u.id ? "Buying…" : "Buy"}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function Guestbook({
  entries,
  hasLedger,
  loading,
}: {
  entries: TavernGuestbookEntry[];
  hasLedger: boolean;
  loading: boolean;
}) {
  if (loading) return <p className="text-sm text-muted animate-pulse">Opening the book…</p>;
  if (!entries.length) return <p className="text-sm text-faint">No names in the book yet.</p>;

  return (
    <div className="space-y-2">
      {!hasLedger && (
        <p className="text-xs text-faint">
          Buy the guest ledger and each visitor gets a history here. The histories are invented;
          the counts and dates below are not.
        </p>
      )}
      {entries.map((e) => (
        <div key={e.species_id} className="fg-card p-3 flex items-start gap-3">
          <span
            className="w-8 h-8 rounded-full shrink-0 border border-line"
            style={{ background: e.style.primary }}
            aria-hidden
          />
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline gap-2 flex-wrap">
              <h3 className="font-serif text-base text-ink">{e.species}</h3>
              <span className="text-[10px] uppercase tracking-wider text-faint">
                {e.archetype.title}
              </span>
              <span className="text-[10px] uppercase tracking-wider text-muted">{e.rarity}</span>
            </div>
            <div className="font-serif italic text-muted text-sm">{e.scientific_name}</div>
            {e.lore && <p className="text-sm text-muted mt-1">{e.lore.backstory}</p>}
            <div className="text-xs text-faint tnum mt-1">
              {e.visits.toLocaleString()} {e.visits === 1 ? "visit" : "visits"}
              {e.times_heard > 0 && ` · heard ${e.times_heard.toLocaleString()}`}
              {e.first_seen && ` · first ${formatDate(e.first_seen)}`}
              {e.last_seen && ` · last ${formatDate(e.last_seen)}`}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Small helpers ───────────────────────────────────────────────────────

function seedLog(patrons: TavernPatron[], events: TavernEvent[]): LogItem[] {
  const items: LogItem[] = [
    ...patrons.map((p) => ({
      kind: "patron" as const,
      id: p.detection_id,
      at: p.arrived_at,
      patron: p,
    })),
    ...events.map((e) => ({
      kind: "event" as const,
      id: e.detection_id,
      at: e.at,
      event: e,
    })),
  ];
  return items.sort((a, b) => b.at.localeCompare(a.at)).slice(0, LOG_LENGTH);
}

/**
 * A clock time, with the date in front of it when the arrival is not from
 * today. On a quiet yard the room falls back to the last company the
 * camera saw, and "11:27 AM" alone reads as this morning when it was May.
 */
function formatTime(iso: string): string {
  const at = new Date(iso);
  const time = at.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const today = new Date();
  const sameDay =
    at.getFullYear() === today.getFullYear() &&
    at.getMonth() === today.getMonth() &&
    at.getDate() === today.getDate();
  return sameDay ? time : `${formatDate(iso)}, ${time}`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" });
}

/**
 * A plucked note per arrival, tuned to the species' own call pitch after
 * the palette has snapped it to a pentatonic scale. Two birds landing at
 * once therefore agree with each other rather than clashing.
 *
 * A minstrel gets a third note on top, which is the closest this gets to
 * the bard the design document asks for.
 */
function useTavernSound(enabled: boolean) {
  const ctxRef = useRef<AudioContext | null>(null);
  const lastRef = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    // The toggle click is the user gesture browsers require before audio.
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!ctxRef.current && Ctor) ctxRef.current = new Ctor();
    void ctxRef.current?.resume();
  }, [enabled]);

  return useCallback(
    (patron: TavernPatron) => {
      const ctx = ctxRef.current;
      if (!enabled || !ctx) return;
      const now = ctx.currentTime;
      // A backlog draining at once would otherwise arrive as one loud chord.
      const at = Math.max(now, lastRef.current + 0.16);
      lastRef.current = at;

      const hz = Math.min(4000, Math.max(140, patron.style.chime_hz || 440));
      const partials: [number, number, number][] =
        patron.archetype.id === "minstrel"
          ? [
              [1, 1, 0],
              [1.5, 0.5, 0.12],
              [2, 0.3, 0.24],
            ]
          : [
              [1, 1, 0],
              [2, 0.28, 0],
            ];

      for (const [mult, level, delay] of partials) {
        const t0 = at + delay;
        const gain = ctx.createGain();
        gain.gain.setValueAtTime(0.0001, t0);
        gain.gain.exponentialRampToValueAtTime(0.07 * level, t0 + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 1.1);
        gain.connect(ctx.destination);

        const osc = ctx.createOscillator();
        // Triangle rather than sine: a lute is plucked, not blown.
        osc.type = "triangle";
        osc.frequency.setValueAtTime(hz * mult, t0);
        osc.connect(gain);
        osc.start(t0);
        osc.stop(t0 + 1.2);
      }
    },
    [enabled]
  );
}
