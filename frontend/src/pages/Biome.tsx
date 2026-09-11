import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { BiomeCanvas, type BiomeCanvasHandle } from "../components/BiomeCanvas";
import {
  CameraIcon,
  ChevronIcon,
  DownloadIcon,
  PauseIcon,
  PlayIcon,
  VolumeIcon,
  VolumeMuteIcon,
} from "../components/FieldIcons";
import {
  fetchBiomeDates,
  fetchBiomeGarden,
  type BiomeGarden,
  type BiomePlant,
} from "../lib/api";
import { useThemeKey } from "../lib/useThemeKey";

/**
 * Chrono-Chirps: the yard's audio as a garden.
 *
 * One plant per species the Haikubox heard, grown from how often and how
 * widely across the day it called. The page owns the day being shown and
 * the clock scrubbing through it; the canvas animates whatever it is
 * handed and knows nothing about fetching.
 *
 * Two honesty rules the page has to keep, because the data does not carry
 * what the design document assumed it would:
 *
 *   - Vitality is call density, not loudness. The Haikubox v2 REST feed
 *     returns neither spectral energy nor a BirdNET score, so the garden
 *     says "call density" out loud rather than implying a microphone
 *     reading it never took.
 *   - A pollinator is either confirmed or coincident, and they look
 *     different. Confirmed means the pipeline matched a sighting to a call
 *     inside the correlation window. Coincident means only that the camera
 *     saw the species the same day.
 */

const DEFAULT_SUN = { sunrise: 0.25, sunset: 0.79 };

/** How long a full sweep of the day's calls takes, in real seconds. */
const SPEEDS = [
  { label: "60s", seconds: 60 },
  { label: "20s", seconds: 20 },
  { label: "7s", seconds: 7 },
];

const FORM_NOTE: Record<BiomePlant["form"], string> = {
  moss: "under 1.2 kHz",
  spire: "1.2 to 2 kHz",
  vine: "2 to 3 kHz",
  frond: "3 to 3.7 kHz",
  floret: "above 3.7 kHz",
};

export default function Biome() {
  const [pickedDate, setPickedDate] = useState<string | null>(null);
  const [hovered, setHovered] = useState<BiomePlant | null>(null);
  const [pinned, setPinned] = useState<BiomePlant | null>(null);
  const [sound, setSound] = useState(false);
  const [pollinators, setPollinators] = useState(true);
  const [exporting, setExporting] = useState(false);

  const canvasRef = useRef<BiomeCanvasHandle | null>(null);
  const themeKey = useThemeKey();

  const datesQ = useQuery({ queryKey: ["biome-dates"], queryFn: () => fetchBiomeDates(60) });
  const dates = datesQ.data?.dates ?? [];
  const date = pickedDate ?? dates[0]?.date ?? null;

  const gardenQ = useQuery({
    queryKey: ["biome-garden", date],
    queryFn: () => fetchBiomeGarden({ date: date!, limit: 40 }),
    enabled: date !== null,
  });

  const garden = gardenQ.data;
  const sun = garden?.sun ?? DEFAULT_SUN;
  const plants = useMemo(() => garden?.plants ?? [], [garden]);

  const clock = useGardenClock(plants, sound);

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
      a.download = `birdwatcher-${date ?? "day"}-biome.png`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }, [date]);

  const inspected = hovered ?? pinned;
  const dateIndex = dates.findIndex((d) => d.date === date);

  return (
    <div className="space-y-4 pb-8">
      <div>
        <div className="fg-overline">Living audio garden</div>
        <h2 className="font-serif font-medium text-2xl text-ink leading-tight mt-0.5">
          Chrono-Chirps
        </h2>
        <p className="text-sm text-muted mt-1">
          What the yard sounded like, grown as a garden. Every species the Haikubox heard puts up
          one plant, and its pitch decides what kind. Deep callers spread as cushion moss on the
          left, thin high whistles open as florets on the right.
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
                {d.date} — {d.call_count} calls, {d.species_count} species
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
        <BiomeCanvas
          ref={canvasRef}
          plants={plants}
          pollinators={garden?.pollinators ?? []}
          sun={sun}
          progressRef={clock.progressRef}
          pickedSpecies={pinned?.species ?? null}
          onHover={setHovered}
          onPick={setPinned}
          themeKey={themeKey}
          showPollinators={pollinators}
        />
        {gardenQ.isFetching && (
          <div className="absolute inset-0 grid place-items-center bg-surface/70 rounded-card">
            <span className="text-xs text-muted animate-pulse">Growing the day…</span>
          </div>
        )}
        {garden?.summary.quiet && !gardenQ.isFetching && (
          <div className="absolute inset-0 grid place-items-center">
            <span className="text-xs text-muted">
              The box heard nothing on this day. Nothing grew.
            </span>
          </div>
        )}
      </div>

      {/* ── Timeline ─────────────────────────────────────────────────── */}
      <div className="fg-card p-3 space-y-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={clock.toggle}
            className="fg-btn-primary px-2.5 py-1.5"
            aria-label={clock.playing ? "Pause" : "Play"}
            title={clock.playing ? "Pause" : "Grow the day"}
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
            Fully grown
          </button>

          <span className="text-sm text-ink tnum font-medium min-w-[5.5rem]">
            {clock.readout < 0 ? "Whole day" : formatClock(clock.readout)}
          </span>

          <div className="ml-auto flex items-center gap-1.5">
            <div
              className="flex items-center gap-0.5 border border-line rounded-lg p-0.5"
              title="Time to grow the whole day"
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
              onClick={() => setPollinators((v) => !v)}
              className={`fg-btn-ghost px-2.5 py-1.5 text-xs ${
                pollinators ? "text-leaf border-leaf" : ""
              }`}
              title="Birds the camera saw as well as the microphone heard"
            >
              Pollinators
            </button>

            <button
              onClick={() => setSound((v) => !v)}
              className={`fg-btn-ghost px-2 py-1.5 ${sound ? "text-leaf border-leaf" : ""}`}
              title={
                sound
                  ? "Silence the garden"
                  : "Sound a note as each species first sings"
              }
              aria-label={sound ? "Mute the garden" : "Unmute the garden"}
            >
              {sound ? <VolumeIcon size={14} /> : <VolumeMuteIcon size={14} />}
            </button>

            <button
              onClick={onExport}
              disabled={exporting || plants.length === 0}
              className="fg-btn-ghost px-2.5 py-1.5 text-xs gap-1.5"
              title="Download this garden at 3840 × 2160"
            >
              <DownloadIcon size={13} />
              <span className="hidden sm:inline">{exporting ? "Rendering…" : "Export"}</span>
            </button>
          </div>
        </div>

        <input
          type="range"
          className="fg-range w-full"
          min={0}
          max={1}
          step={0.0005}
          value={clock.readout < 0 ? 1 : clock.readout}
          onChange={(e) => clock.seek(parseFloat(e.target.value))}
          aria-label="Time of day"
        />
        {garden && <Chorus chorus={garden.chorus} sun={sun} />}
      </div>

      {inspected && <PlantCard plant={inspected} onClear={() => setPinned(null)} />}

      {garden && !garden.summary.quiet && <Reading garden={garden} />}

      {garden && plants.length > 0 && (
        <div className="space-y-2">
          <div className="fg-overline">The bed</div>
          <ul className="space-y-1.5">
            {[...plants]
              .sort((a, b) => b.calls - a.calls)
              .map((p) => (
                <li key={p.species}>
                  <button
                    onClick={() => setPinned(pinned?.species === p.species ? null : p)}
                    className={`w-full fg-card p-2.5 flex items-center gap-2.5 text-left transition-colors ${
                      pinned?.species === p.species ? "border-leaf" : ""
                    }`}
                  >
                    <span
                      className="w-2.5 h-2.5 rounded-full shrink-0"
                      style={{
                        background: `hsl(${p.bloom.hue}deg ${Math.round(
                          p.bloom.sat * 100
                        )}% ${Math.round(p.bloom.light * 100)}%)`,
                      }}
                      aria-hidden
                    />
                    <span className="min-w-0 flex-1">
                      <span className="text-sm text-ink">{p.species}</span>
                      <span className="text-xs text-faint"> · {p.form_title}</span>
                      {p.flowering && (
                        <span className="inline-flex items-center gap-1 text-[10px] text-leaf uppercase tracking-wider ml-2">
                          <CameraIcon size={11} />
                          {p.confirmed ? "Seen and heard" : "Seen that day"}
                        </span>
                      )}
                    </span>
                    <span className="text-xs text-faint tnum shrink-0">
                      {p.calls.toLocaleString()} {p.calls === 1 ? "call" : "calls"}
                    </span>
                  </button>
                </li>
              ))}
          </ul>
        </div>
      )}

      {gardenQ.error && (
        <p className="text-sm text-rust">The garden is not answering. The API may be down.</p>
      )}
    </div>
  );
}

// ── Pieces ──────────────────────────────────────────────────────────────

/** The whole yard's calls per hour, under the scrubber it belongs to. */
function Chorus({ chorus, sun }: { chorus: number[]; sun: { sunrise: number; sunset: number } }) {
  const peak = Math.max(1, ...chorus);
  return (
    <div>
      <div className="flex items-end gap-[2px] h-8" aria-hidden>
        {chorus.map((n, hour) => {
          const day = hour / 24 >= sun.sunrise && hour / 24 <= sun.sunset;
          return (
            <div
              key={hour}
              className={`flex-1 rounded-sm ${day ? "bg-leaf" : "bg-line"}`}
              style={{ height: `${Math.max(2, (n / peak) * 100)}%`, opacity: n ? 1 : 0.35 }}
              title={`${hour}:00 — ${n} ${n === 1 ? "call" : "calls"}`}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-[11px] text-faint tnum mt-1">
        <span>Midnight</span>
        <span>Sunrise {formatClock(sun.sunrise)}</span>
        <span>Sunset {formatClock(sun.sunset)}</span>
        <span>Midnight</span>
      </div>
    </div>
  );
}

/** What the day's numbers were, and what they are not. */
function Reading({ garden }: { garden: BiomeGarden }) {
  const { summary } = garden;
  const confirmed = garden.pollinators.filter((p) => p.confirmed).length;
  const coincident = garden.pollinators.length - confirmed;

  return (
    <div className="fg-card p-3 text-sm text-muted space-y-1.5">
      <p>
        <span className="text-ink font-semibold tnum">{summary.calls.toLocaleString()}</span> calls
        from <span className="text-ink font-semibold tnum">{summary.species_heard}</span>{" "}
        {summary.species_heard === 1 ? "species" : "species"}
        {summary.peak_hour !== null && (
          <>
            , loudest at{" "}
            <span className="text-ink font-semibold tnum">
              {formatClock(summary.peak_hour / 24)}
            </span>
          </>
        )}
        .{summary.truncated && ` The ${summary.returned} most-heard are planted.`}
      </p>
      <p className="text-xs text-faint">
        How alive a plant looks comes from how often the species called and how much of the day it
        called across, not from how loud it was. The Haikubox REST feed gives neither spectral
        energy nor a confidence score, so nothing here is a loudness reading.
      </p>
      {garden.pollinators.length > 0 ? (
        <p className="text-xs text-faint">
          Pollinators: <span className="text-leaf tnum">{confirmed}</span> seen and heard inside the
          correlation window, <span className="tnum">{coincident}</span> seen by the camera only
          somewhere in the same day. The second is the weaker claim and flies paler.
        </p>
      ) : (
        <p className="text-xs text-faint">
          Nothing pollinated today: the camera identified none of the species the box heard.
        </p>
      )}
    </div>
  );
}

function PlantCard({ plant, onClear }: { plant: BiomePlant; onClear: () => void }) {
  const bloom = `hsl(${plant.bloom.hue}deg ${Math.round(plant.bloom.sat * 100)}% ${Math.round(
    plant.bloom.light * 100
  )}%)`;
  return (
    <div className="fg-card p-3 flex gap-3">
      <div
        className="w-16 h-16 rounded-lg shrink-0 border border-line"
        style={{
          background: `linear-gradient(160deg, ${bloom}, ${plant.plumage.primary})`,
        }}
        aria-hidden
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap">
          <h3 className="font-serif text-lg text-ink leading-tight">{plant.species}</h3>
          <span className="text-[10px] uppercase tracking-wider text-faint">
            {plant.form_title}
          </span>
          {plant.flowering && (
            <span className="inline-flex items-center gap-1 text-[10px] text-leaf uppercase tracking-wider">
              <CameraIcon size={11} /> {plant.confirmed ? "Seen and heard" : "Seen that day"}
            </span>
          )}
        </div>
        <p className="text-sm text-muted mt-0.5">{plant.form_blurb}</p>
        <div className="text-xs text-faint tnum mt-1">
          {plant.calls.toLocaleString()} {plant.calls === 1 ? "call" : "calls"}
          {" · "}
          {Math.round(plant.share * 100)}% of the day
          {" · "}
          {formatClock(plant.first_heard)} to {formatClock(plant.last_heard)}
        </div>
        <div className="text-xs text-faint tnum">
          {(plant.call_hz / 1000).toFixed(1)} kHz, {FORM_NOTE[plant.form]}
          {plant.mean_confidence !== null && ` · score ${plant.mean_confidence.toFixed(2)}`}
        </div>
      </div>
      <button className="fg-btn-ghost px-2 py-1 text-xs self-start" onClick={onClear}>
        Clear
      </button>
    </div>
  );
}

// ── The clock ───────────────────────────────────────────────────────────

/**
 * Drive the day's growth outside React.
 *
 * Same argument the Art page's clock makes: the position advances every
 * animation frame, and holding it in state would re-render this page and
 * its canvas sixty times a second. It lives in a ref the canvas reads, and
 * a throttled copy reaches state so the digits above the scrubber tick.
 *
 * The sweep runs from the day's first call to its last rather than from
 * sunrise to sunset. An owl at two in the morning is exactly the thing
 * worth watching grow, and a sunrise-bounded sweep would skip it.
 */
function useGardenClock(plants: BiomePlant[], sound: boolean) {
  const progressRef = useRef(-1);
  const [readout, setReadout] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [sweep, setSweep] = useState(20);

  // Synced in an effect, not during render: writing a ref while rendering
  // is not safe once React can render a tree it later throws away.
  const plantsRef = useRef(plants);
  useEffect(() => {
    plantsRef.current = plants;
  }, [plants]);

  const note = useGardenVoice(sound);

  const first = plants.length ? Math.min(...plants.map((p) => p.first_heard)) : 0;
  const last = plants.length ? Math.max(...plants.map((p) => p.last_heard)) : 1;
  const from0 = Math.max(0, first - 0.02);
  const to1 = Math.min(1, last + 0.02);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let prev = performance.now();
    let lastReadout = 0;

    const step = (now: number) => {
      const dt = Math.min(0.1, (now - prev) / 1000);
      prev = now;

      const from = progressRef.current < 0 ? from0 : progressRef.current;
      const to = from + (dt / sweep) * (to1 - from0);

      // One note per species, the first time it sings. A note per call
      // would be thousands of them, and the garden is meant to be calm.
      const waking = plantsRef.current.filter(
        (p) => p.first_heard > from && p.first_heard <= to
      );
      for (const p of waking.slice(0, 3)) note(p.chime_hz);

      progressRef.current = to;
      if (now - lastReadout > 120) {
        lastReadout = now;
        setReadout(to);
      }
      if (to >= to1) {
        progressRef.current = to1;
        setReadout(to1);
        setPlaying(false);
        return;
      }
      raf = requestAnimationFrame(step);
    };

    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, sweep, from0, to1, note]);

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
      if (!p && (progressRef.current < 0 || progressRef.current >= to1)) {
        progressRef.current = from0;
        setReadout(from0);
      }
      return !p;
    });
  }, [from0, to1]);

  return { progressRef, readout, playing, sweep, setSweep, seek, showWholeDay, toggle };
}

/**
 * A soft sustained note as a species first sings.
 *
 * A sine with a slow attack and a long tail, plus a quiet fifth above it,
 * which is about as close to a flute as two oscillators get. Pitches come
 * from the backend already snapped to a pentatonic scale, so two species
 * waking together agree with each other.
 *
 * This is not the bird. Playing the real recording would mean the Haikubox
 * AppSync API and its presigned FLAC URLs, whose credentials this repo
 * does not have, so the garden hums rather than pretending.
 */
function useGardenVoice(enabled: boolean) {
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
    (hz: number) => {
      const ctx = ctxRef.current;
      if (!enabled || !ctx) return;
      const now = ctx.currentTime;
      // Species waking within a few frames of each other would otherwise
      // land as one chord loud enough to startle.
      const at = Math.max(now, lastRef.current + 0.25);
      lastRef.current = at;

      const pitch = Math.min(3200, Math.max(120, hz || 440));
      for (const [mult, level] of [
        [1, 1],
        [1.5, 0.35],
      ] as [number, number][]) {
        const gain = ctx.createGain();
        gain.gain.setValueAtTime(0.0001, at);
        gain.gain.exponentialRampToValueAtTime(0.055 * level, at + 0.35);
        gain.gain.exponentialRampToValueAtTime(0.0001, at + 2.6);
        gain.connect(ctx.destination);

        const osc = ctx.createOscillator();
        osc.type = "sine";
        osc.frequency.setValueAtTime(pitch * mult, at);
        osc.connect(gain);
        osc.start(at);
        osc.stop(at + 2.7);
      }
    },
    [enabled]
  );
}

function formatClock(fraction: number): string {
  const minutes = Math.round(fraction * 24 * 60);
  const h24 = Math.floor(minutes / 60) % 24;
  const m = minutes % 60;
  const suffix = h24 >= 12 ? "pm" : "am";
  const h12 = h24 % 12 === 0 ? 12 : h24 % 12;
  return `${h12}:${String(m).padStart(2, "0")} ${suffix}`;
}
