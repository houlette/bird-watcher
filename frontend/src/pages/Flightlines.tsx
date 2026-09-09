import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { ArtCanvas, type ArtCanvasHandle, type ArtMode } from "../components/ArtCanvas";
import {
  ChevronIcon,
  DownloadIcon,
  PauseIcon,
  PlayIcon,
  SoundIcon,
  VolumeIcon,
  VolumeMuteIcon,
} from "../components/FieldIcons";
import { fetchArtDates, fetchArtDay, type ArtDay, type ArtFlight, type ArtSun } from "../lib/api";

const DEFAULT_SUN: ArtSun = { sunrise: 0.25, sunset: 0.79 };

/** How long a full dawn-to-dusk sweep takes, in real seconds. */
const SPEEDS = [
  { label: "60s", seconds: 60 },
  { label: "20s", seconds: 20 },
  { label: "7s", seconds: 7 },
];

const MODES: { id: ArtMode; label: string; caption: string }[] = [
  {
    id: "flightlines",
    label: "Flightlines",
    caption:
      "Every flight as an ink ribbon over the camera's field of view. The stroke widens where the bird slowed down, so the heaviest mark falls on the perch.",
  },
  {
    id: "mandala",
    label: "Celestial mandala",
    caption:
      "A 24-hour dial, midnight at the top, running clockwise. Each bird is a bead stacked outward from the hub in its quarter-hour, so a busy dawn grows a long spoke. Rings mark every five birds; the shaded arc is the hours after sunset.",
  },
  {
    id: "topography",
    label: "Topography",
    caption:
      "Contours of where birds actually chose to stop. Sand piles up over the busy perches and takes on the plumage of whatever lands there.",
  },
];

export default function Flightlines() {
  const [mode, setMode] = useState<ArtMode>("flightlines");
  const [pickedDate, setPickedDate] = useState<string | null>(null);
  const [species, setSpecies] = useState<string | null>(null);
  const [showUnidentified, setShowUnidentified] = useState(true);
  const [hovered, setHovered] = useState<ArtFlight | null>(null);
  const [pinned, setPinned] = useState<ArtFlight | null>(null);
  const [sound, setSound] = useState(false);
  const [exporting, setExporting] = useState(false);

  const canvasRef = useRef<ArtCanvasHandle | null>(null);
  const themeKey = useThemeKey();

  const datesQ = useQuery({ queryKey: ["art-dates"], queryFn: () => fetchArtDates(60) });
  const dates = datesQ.data?.dates ?? [];
  const date = pickedDate ?? dates[0]?.date ?? null;

  const dayQ = useQuery({
    queryKey: ["art-day", date, showUnidentified],
    queryFn: () => fetchArtDay({ date: date!, limit: 160, include_unidentified: showUnidentified }),
    enabled: date !== null,
  });

  const day = dayQ.data;
  const sun = day?.sun ?? DEFAULT_SUN;

  // Species filtering happens here rather than server-side so the chips
  // respond instantly and don't re-fetch a day the browser already holds.
  const flights = useMemo(() => {
    const all = day?.flights ?? [];
    return species ? all.filter((f) => f.species === species) : all;
  }, [day, species]);

  const clock = useDayClock(sun, flights, sound);

  // Choosing a new day resets the timeline and clears any pinned specimen.
  // Done here rather than in an effect on `date`: the only other thing that
  // changes `date` is the first load settling on the newest day, where
  // there is nothing to reset, and resetting from an effect would mean a
  // second render pass on every day change.
  const chooseDate = useCallback(
    (next: string) => {
      setPickedDate(next);
      clock.showWholeDay();
      setPinned(null);
      setHovered(null);
    },
    [clock]
  );

  const onExport = useCallback(async () => {
    setExporting(true);
    try {
      const blob = await canvasRef.current?.exportPNG(3840);
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `birdwatcher-${date ?? "day"}-${mode}.png`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }, [date, mode]);

  const activeMode = MODES.find((m) => m.id === mode)!;
  const inspected = hovered ?? pinned;
  const dateIndex = dates.findIndex((d) => d.date === date);

  return (
    <div className="space-y-4 pb-8">
      <div>
        <div className="fg-overline">Generative studio</div>
        <h2 className="font-serif font-medium text-2xl text-ink leading-tight mt-0.5">
          Flightlines
        </h2>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1 p-1 rounded-lg bg-panel border border-line">
          {MODES.map((m) => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                mode === m.id ? "bg-surface text-leaf font-semibold" : "text-muted hover:text-ink"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

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
                {d.date} — {d.flight_count} flights
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
        <ArtCanvas
          ref={canvasRef}
          flights={flights}
          mode={mode}
          sun={sun}
          progressRef={clock.progressRef}
          hoveredId={hovered?.detection_id ?? null}
          pinnedId={pinned?.detection_id ?? null}
          onHover={setHovered}
          onPick={setPinned}
          themeKey={themeKey}
        />
        {dayQ.isFetching && (
          <div className="absolute inset-0 grid place-items-center bg-surface/70 rounded-card">
            <span className="text-xs text-muted animate-pulse">Drawing the day…</span>
          </div>
        )}
      </div>

      <p className="text-xs text-faint">{activeMode.caption}</p>

      {/* ── Timeline ─────────────────────────────────────────────────── */}
      <div className="fg-card p-3 space-y-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={clock.toggle}
            className="fg-btn-primary px-2.5 py-1.5"
            aria-label={clock.playing ? "Pause" : "Play"}
            title={clock.playing ? "Pause" : "Play the day"}
          >
            {clock.playing ? <PauseIcon size={14} /> : <PlayIcon size={14} />}
          </button>

          <button
            onClick={clock.showWholeDay}
            className={`px-2.5 py-1.5 rounded-lg text-xs border transition-colors ${
              clock.readout < 0
                ? "border-leaf text-leaf font-semibold"
                : "border-line text-muted hover:text-ink"
            }`}
          >
            Whole day
          </button>

          <span className="text-sm text-ink tnum font-medium min-w-[5.5rem]">
            {clock.readout < 0 ? "All day" : formatClock(clock.readout)}
          </span>

          <div className="ml-auto flex items-center gap-1.5">
            <div
              className="flex items-center gap-0.5 border border-line rounded-lg p-0.5"
              title="Time to sweep the whole day"
            >
              {SPEEDS.map((s) => (
                <button
                  key={s.label}
                  onClick={() => clock.setSweep(s.seconds)}
                  className={`px-1.5 py-0.5 text-[11px] rounded tnum ${
                    clock.sweep === s.seconds
                      ? "bg-leaf text-surface font-semibold"
                      : "text-muted hover:text-ink"
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>

            <button
              onClick={() => setSound((v) => !v)}
              className={`fg-btn-ghost px-2 py-1.5 ${sound ? "text-leaf border-leaf" : ""}`}
              title={sound ? "Mute arrival chimes" : "Chime as each bird arrives"}
              aria-label={sound ? "Mute chimes" : "Enable chimes"}
            >
              {sound ? <VolumeIcon size={14} /> : <VolumeMuteIcon size={14} />}
            </button>

            <button
              onClick={onExport}
              disabled={exporting || flights.length === 0}
              className="fg-btn-ghost px-2.5 py-1.5 text-xs gap-1.5"
              title="Download this artwork at 3840 × 2160"
            >
              <DownloadIcon size={13} />
              <span className="hidden sm:inline">{exporting ? "Rendering…" : "Export"}</span>
            </button>
          </div>
        </div>

        <input
          type="range"
          className="fg-range w-full"
          min={sun.sunrise - 0.03}
          max={sun.sunset + 0.03}
          step={0.0005}
          value={clock.readout < 0 ? sun.sunset + 0.03 : clock.readout}
          onChange={(e) => clock.seek(parseFloat(e.target.value))}
          aria-label="Time of day"
        />
        <div className="flex justify-between text-[11px] text-faint tnum">
          <span>Sunrise {formatClock(sun.sunrise)}</span>
          <span>Sunset {formatClock(sun.sunset)}</span>
        </div>
      </div>

      {/* ── Species ──────────────────────────────────────────────────── */}
      {day && (
        <div className="space-y-2">
          <div className="fg-overline">Species</div>
          <div className="flex flex-wrap gap-1.5">
            <Chip active={species === null} onClick={() => setSpecies(null)}>
              All ({day.summary.returned})
            </Chip>
            {Object.entries(day.summary.species_counts).map(([name, count]) => (
              <Chip
                key={name}
                active={species === name}
                color={day.summary.species_colors[name]}
                onClick={() => setSpecies(species === name ? null : name)}
              >
                {name} <span className="text-faint tnum">{count}</span>
              </Chip>
            ))}
          </div>
          <label className="flex items-center gap-2 text-xs text-muted pt-0.5">
            <input
              type="checkbox"
              className="fg-range"
              checked={showUnidentified}
              onChange={(e) => setShowUnidentified(e.target.checked)}
            />
            Include crops the classifier rejected
          </label>
        </div>
      )}

      {/* ── Inspector ────────────────────────────────────────────────── */}
      {inspected && (
        <div className="fg-card p-3 flex items-start gap-3">
          {inspected.crop_url ? (
            <img
              src={inspected.crop_url}
              alt={inspected.species}
              className="w-16 h-16 rounded-lg object-cover border border-line shrink-0"
            />
          ) : (
            <div
              className="w-16 h-16 rounded-lg shrink-0"
              style={{ background: inspected.style.primary }}
            />
          )}
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-serif text-lg text-ink leading-tight">{inspected.species}</h3>
              {inspected.audio_confirmed && (
                <span className="inline-flex items-center gap-1 text-[10px] text-leaf uppercase tracking-wider">
                  <SoundIcon size={11} /> Heard too
                </span>
              )}
            </div>
            {inspected.scientific_name && (
              <div className="font-serif italic text-muted text-sm">
                {inspected.scientific_name}
              </div>
            )}
            <div className="text-xs text-faint tnum mt-1">
              {new Date(inspected.started_at).toLocaleTimeString([], {
                hour: "numeric",
                minute: "2-digit",
              })}
              {" · "}
              {inspected.path_kind === "tracked"
                ? "path from per-frame tracking"
                : "perch real, approach synthesized"}
            </div>
          </div>
          {pinned && (
            <button className="fg-btn-ghost px-2 py-1 text-xs" onClick={() => setPinned(null)}>
              Clear
            </button>
          )}
        </div>
      )}

      {day && <Provenance day={day} flights={flights} />}

      {dayQ.error && (
        <p className="text-sm text-rust">
          Couldn't load the day: {(dayQ.error as Error).message}
        </p>
      )}
    </div>
  );
}

function Chip({
  active,
  color,
  onClick,
  children,
}: {
  active: boolean;
  color?: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border transition-colors ${
        active
          ? "border-leaf text-ink bg-surface font-semibold"
          : "border-line text-muted bg-surface hover:text-ink"
      }`}
    >
      {color && (
        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
      )}
      {children}
    </button>
  );
}

/**
 * Say where the picture came from.
 *
 * Almost every row in the database predates per-frame tracking, so almost
 * every ribbon on the canvas is a real perch with an invented approach.
 * That is a fine thing for an artwork to do and a bad thing for it to hide.
 */
function Provenance({ day, flights }: { day: ArtDay; flights: ArtFlight[] }) {
  const tracked = flights.filter((f) => f.path_kind === "tracked").length;
  const total = day.summary.total_available;

  const paths =
    tracked === 0
      ? "Every path places the bird on the perch the camera saw and invents the approach and departure."
      : tracked === flights.length
      ? "Every path follows the bird frame by frame."
      : `${tracked} of them follow the bird frame by frame; the rest place it on the perch the camera saw and invent the approach and departure.`;

  return (
    <p className="text-xs text-faint">
      Drawing {flights.length} of the day's {total} flight{total === 1 ? "" : "s"}
      {day.summary.truncated
        ? ", sampled evenly across the day so the artwork still spans dawn to dusk"
        : ""}
      . {paths} Times are the feeder's own clock ({day.tz}).
    </p>
  );
}

// ── Timeline clock ──────────────────────────────────────────────────────

/**
 * Drive playback outside React.
 *
 * The position advances every animation frame; storing it in state would
 * re-render this page and its canvas sixty times a second. It lives in a
 * ref the canvas reads directly, and only a throttled copy reaches state,
 * purely so the digits above the scrubber tick over.
 */
function useDayClock(sun: ArtSun, flights: ArtFlight[], sound: boolean) {
  const progressRef = useRef(-1);
  const [readout, setReadout] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [sweep, setSweep] = useState(20);

  // Synced in an effect, not during render: writing a ref while rendering
  // is not safe once React can render a tree it later throws away.
  const flightsRef = useRef(flights);
  useEffect(() => {
    flightsRef.current = flights;
  }, [flights]);

  const chime = useChimes(sound);

  const dawn = sun.sunrise - 0.03;
  const dusk = sun.sunset + 0.03;

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    let lastReadout = 0;

    const step = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;

      const from = progressRef.current < 0 ? dawn : progressRef.current;
      const to = from + (dt / sweep) * (dusk - dawn);

      // Anything that arrived in the slice we just crossed gets a note.
      // Capped so a burst of pigeons doesn't turn into a machine-gun.
      const arrivals = flightsRef.current.filter(
        (f) => f.time_of_day > from && f.time_of_day <= to
      );
      for (const f of arrivals.slice(0, 3)) chime(f.style.chime_hz);

      progressRef.current = to;
      if (now - lastReadout > 120) {
        lastReadout = now;
        setReadout(to);
      }
      if (to >= dusk) {
        progressRef.current = dusk;
        setReadout(dusk);
        setPlaying(false);
        return;
      }
      raf = requestAnimationFrame(step);
    };

    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, sweep, dawn, dusk, chime]);

  const seek = useCallback((v: number) => {
    setPlaying(false);
    progressRef.current = v;
    setReadout(v);
  }, []);

  const showWholeDay = useCallback(() => {
    setPlaying(false);
    progressRef.current = -1;
    setReadout(-1);
  }, []);

  const toggle = useCallback(() => {
    setPlaying((p) => {
      if (!p && (progressRef.current < 0 || progressRef.current >= dusk)) {
        progressRef.current = dawn;
        setReadout(dawn);
      }
      return !p;
    });
  }, [dawn, dusk]);

  return { progressRef, readout, playing, sweep, setSweep, seek, showWholeDay, toggle };
}

/**
 * Short bell tones for arrivals.
 *
 * A fundamental plus a quiet octave through a fast decay reads as a chime
 * rather than a beep. Pitches arrive pre-quantised to a pentatonic scale
 * from the backend, so overlapping arrivals stay consonant.
 */
function useChimes(enabled: boolean) {
  const ctxRef = useRef<AudioContext | null>(null);
  const lastRef = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    // The toggle click is the user gesture browsers require before audio.
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!ctxRef.current && Ctor) ctxRef.current = new Ctor();
    ctxRef.current?.resume();
  }, [enabled]);

  return useCallback(
    (hz: number) => {
      const ctx = ctxRef.current;
      if (!enabled || !ctx) return;
      const now = ctx.currentTime;
      if (now - lastRef.current < 0.07) return;
      lastRef.current = now;

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.09, now + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.9);
      gain.connect(ctx.destination);

      for (const [mult, level] of [
        [1, 1],
        [2, 0.35],
      ] as const) {
        const osc = ctx.createOscillator();
        osc.type = "sine";
        osc.frequency.setValueAtTime(Math.min(5000, Math.max(120, hz)) * mult, now);
        const sub = ctx.createGain();
        sub.gain.value = level;
        osc.connect(sub).connect(gain);
        osc.start(now);
        osc.stop(now + 1);
      }
    },
    [enabled]
  );
}

// ── Small helpers ───────────────────────────────────────────────────────

/** Re-read the palette whenever the Sage/Twilight class flips on <html>. */
function useThemeKey() {
  const [key, setKey] = useState(() =>
    typeof document === "undefined" ? "" : document.documentElement.className
  );
  useEffect(() => {
    const ob = new MutationObserver(() => setKey(document.documentElement.className));
    ob.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => ob.disconnect();
  }, []);
  return key;
}

/** A day fraction as a wall-clock time at the feeder. */
function formatClock(fraction: number): string {
  const minutes = Math.round(fraction * 24 * 60);
  const h24 = Math.floor(minutes / 60) % 24;
  const m = minutes % 60;
  const suffix = h24 >= 12 ? "pm" : "am";
  const h12 = h24 % 12 === 0 ? 12 : h24 % 12;
  return `${h12}:${String(m).padStart(2, "0")} ${suffix}`;
}
