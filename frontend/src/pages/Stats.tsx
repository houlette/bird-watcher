import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fetchStats, type DailyStats, type StatsResponse } from "../lib/api";

// Tailwind palette echoes (forest/cream/slate/sand). Recharts wants raw
// CSS colors; keeping them inline so this file is self-contained.
const COLOR_FOREST = "#2f5d3d";
const COLOR_SAND = "#d4b483";
const COLOR_SLATE = "#64748b";
const COLOR_RED = "#dc2626";
const COLOR_BLUE = "#2563eb";

// ─── Model-update markers ────────────────────────────────────────────────
//
// Vertical reference lines on the time-series charts mark days when we
// swapped a model in production, so an accuracy / FP-rate inflection
// can be read against the cause rather than misread as a data anomaly.
// Add new entries here (UTC date of the deploy) as we ship more model
// updates. Matched against the formatted x-axis tick — `date` MUST be
// the same string `shortDate()` produces, i.e. "Jun 7".
type ModelMarker = { date: string; label: string };
const MODEL_MARKERS: ModelMarker[] = [
  { date: "Jun 7", label: "NAB filter" },
  { date: "Jun 8", label: "birdclass-na" },
];

// ─── Helpers ──────────────────────────────────────────────────────────────

function fmtPct(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${Math.round(x * 100)}%`;
}

function shortDate(iso: string): string {
  // "2026-05-30" → "May 30". Tick labels.
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

// ─── Top-level cards ──────────────────────────────────────────────────────

function HeadlineCards({ data }: { data: StatsResponse }) {
  const today = data.daily[data.daily.length - 1];
  // The "yesterday delta" comparison uses the 2nd-to-last row when present.
  const yesterday = data.daily.length >= 2 ? data.daily[data.daily.length - 2] : null;
  const corrections30d = data.daily.reduce(
    (s, d) => s + d.detections_user_corrected,
    0,
  );

  const cards: { label: string; value: string; sub?: string }[] = [
    {
      label: "Clips today",
      value: String(today.clips_received),
      sub: yesterday ? `yesterday ${yesterday.clips_received}` : undefined,
    },
    {
      label: "Detections today",
      value: String(today.detections_total),
      sub: yesterday ? `yesterday ${yesterday.detections_total}` : undefined,
    },
    {
      label: "Corrections (30d)",
      value: String(corrections30d),
      sub: `${data.totals.corrections_total} all-time`,
    },
    {
      label: "Label backlog",
      value: String(data.totals.pending_backlog),
      sub: "model-labeled, awaiting you",
    },
    {
      label: "Ready to fine-tune",
      value: String(data.totals.ready_to_fine_tune_species),
      sub: "species ≥ 50 labels",
    },
    {
      label: "Mask-suppressed today",
      value: String(today.detections_scene_mask_suppressed ?? 0),
      sub: "YOLO hits dropped by hot-zone mask",
    },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      {cards.map((c) => (
        <div key={c.label} className="bg-white rounded-lg shadow-sm p-3">
          <div className="text-xs text-slate-500">{c.label}</div>
          <div className="text-2xl font-semibold text-forest">{c.value}</div>
          {c.sub && <div className="text-xs text-slate-400">{c.sub}</div>}
        </div>
      ))}
    </div>
  );
}

// ─── Funnel chart ─────────────────────────────────────────────────────────

function FunnelChart({ daily }: { daily: DailyStats[] }) {
  const rows = useMemo(
    () =>
      daily.map((d) => ({
        date: shortDate(d.date),
        Clips: d.clips_received,
        Detections: d.detections_total,
        "Classifier-labeled": d.detections_labeled_by_classifier,
        Corrected: d.detections_user_corrected,
        "Mask-suppressed": d.detections_scene_mask_suppressed ?? 0,
      })),
    [daily],
  );

  return (
    <section className="bg-white rounded-lg shadow-sm p-3">
      <h3 className="text-sm font-semibold mb-2">Pipeline funnel (30d)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows} margin={{ left: -10, right: 10, top: 4, bottom: 4 }}>
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {MODEL_MARKERS.map((m) => (
            <ReferenceLine
              key={m.date}
              x={m.date}
              stroke={COLOR_SLATE}
              strokeDasharray="2 3"
              label={{ value: m.label, position: "top", fontSize: 10, fill: COLOR_SLATE }}
            />
          ))}
          <Line type="monotone" dataKey="Clips" stroke={COLOR_SLATE} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Detections" stroke={COLOR_FOREST} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Classifier-labeled" stroke={COLOR_BLUE} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Corrected" stroke={COLOR_SAND} strokeWidth={2} dot={false} />
          {/* Dashed so the line reads as "diagnostic" rather than a
              first-class funnel stage. A spike here means lots of
              motion is being silently filtered — and may include real
              birds that landed in NAB-clustered regions. */}
          <Line
            type="monotone"
            dataKey="Mask-suppressed"
            stroke={COLOR_RED}
            strokeWidth={2}
            strokeDasharray="4 3"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}

// ─── Rate chart (the four headline ratios) ────────────────────────────────

function RatesChart({ daily }: { daily: DailyStats[] }) {
  const rows = useMemo(
    () =>
      daily.map((d) => ({
        date: shortDate(d.date),
        // Recharts handles null by leaving a gap — preferred over 0
        // because "no data" ≠ "zero rate."
        "YOLO bird rate": d.yolo_bird_rate,
        "Classifier label rate": d.classifier_label_rate,
        "User-FP rate": d.user_fp_rate,
        "Classifier accuracy": d.classifier_accuracy,
      })),
    [daily],
  );

  return (
    <section className="bg-white rounded-lg shadow-sm p-3">
      <h3 className="text-sm font-semibold mb-2">Pipeline rates (30d)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows} margin={{ left: -10, right: 10, top: 4, bottom: 4 }}>
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <Tooltip formatter={(v) => (v == null ? "—" : fmtPct(Number(v)))} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {MODEL_MARKERS.map((m) => (
            <ReferenceLine
              key={m.date}
              x={m.date}
              stroke={COLOR_SLATE}
              strokeDasharray="2 3"
              label={{ value: m.label, position: "top", fontSize: 10, fill: COLOR_SLATE }}
            />
          ))}
          <Line type="monotone" dataKey="YOLO bird rate" stroke={COLOR_FOREST} strokeWidth={2} dot={false} connectNulls />
          <Line type="monotone" dataKey="Classifier label rate" stroke={COLOR_BLUE} strokeWidth={2} dot={false} connectNulls />
          <Line type="monotone" dataKey="User-FP rate" stroke={COLOR_RED} strokeWidth={2} dot={false} connectNulls />
          <Line type="monotone" dataKey="Classifier accuracy" stroke={COLOR_SAND} strokeWidth={2} dot={false} connectNulls />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-xs text-slate-500 mt-1">
        Gaps mean no data that day (e.g., no clips → no rates to compute).
      </p>
    </section>
  );
}

// ─── Per-species accuracy bar chart ──────────────────────────────────────

function SpeciesAccuracy({ totals }: { totals: StatsResponse["totals"] }) {
  const rows = totals.species_accuracy.map((s) => ({
    species: s.species,
    accuracy: s.accuracy,
    n: s.n,
  }));
  if (rows.length === 0) {
    return (
      <section className="bg-white rounded-lg shadow-sm p-3">
        <h3 className="text-sm font-semibold mb-2">Per-species accuracy</h3>
        <p className="text-xs text-slate-500">No species has ≥ 5 ground-truth labels yet.</p>
      </section>
    );
  }
  return (
    <section className="bg-white rounded-lg shadow-sm p-3">
      <h3 className="text-sm font-semibold mb-2">
        Classifier vs corrections <span className="text-xs text-slate-400 font-normal">(top 15 by label count)</span>
      </h3>
      <p className="text-xs text-slate-500 mb-2">
        Of the times you labeled a detection as species X, how often the classifier
        had already guessed X (top-1). Biased down by the fact that you mostly correct
        mistakes — low bars don't mean the model is hopeless, they mean you only see it
        when it's wrong.
      </p>
      <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 24)}>
        <BarChart data={rows} layout="vertical" margin={{ left: 10, right: 30, top: 4, bottom: 4 }}>
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <YAxis type="category" dataKey="species" tick={{ fontSize: 11 }} width={140} />
          <Tooltip formatter={(v, _name, p) => [`${fmtPct(Number(v))} of ${p.payload.n}`, "Top-1 accuracy"]} />
          {/* Single color: the bar length carries the signal. Threshold-
              based coloring was misleading here because the expected
              value is near zero (correction-bias). */}
          <Bar dataKey="accuracy" fill={COLOR_FOREST} radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}

// ─── Training-data progress (per-species, by trust tier) ────────────────

// Tier colors. Cream for the highest-trust GOLD tier (matches the
// app's accent palette so it reads "this is the good stuff"); sand for
// HIGH (LLM auto-committed but unreviewed); slate-300 for MED (auto-
// committed, sitting in the review queue and worth burning down).
const TIER_COLOR_GOLD = "#2f5d3d"; // forest
const TIER_COLOR_HIGH = "#d4b483"; // sand
const TIER_COLOR_MED = "#cbd5e1";  // slate-300

function TrainingDataCard({ totals }: { totals: StatsResponse["totals"] }) {
  const rows = totals.training_data ?? [];
  if (rows.length === 0) {
    return (
      <section className="bg-white rounded-lg shadow-sm p-3">
        <h3 className="text-sm font-semibold mb-2">Training data progress</h3>
        <p className="text-xs text-slate-500">No labeled species yet.</p>
      </section>
    );
  }
  // Find a sane x-axis cap so the long tail doesn't crush the big bars
  // to invisible widths. Use 1.05× the largest total, rounded up to the
  // next 100. (Without this, NAB at ~5k dwarfs everything.)
  const maxTotal = Math.max(...rows.map((r) => r.total));
  const xMax = Math.ceil((maxTotal * 1.05) / 100) * 100;

  return (
    <section className="bg-white rounded-lg shadow-sm p-3">
      <h3 className="text-sm font-semibold mb-1">
        Training data progress{" "}
        <span className="text-xs text-slate-400 font-normal">(top 25 by label count)</span>
      </h3>
      <p className="text-xs text-slate-500 mb-2">
        Per-species labeled detections, stacked by trust tier.{" "}
        <span className="font-medium" style={{ color: TIER_COLOR_GOLD }}>Gold</span> = you verified;{" "}
        <span className="font-medium" style={{ color: TIER_COLOR_HIGH }}>High</span> = Claude HIGH (auto-committed, unreviewed);{" "}
        <span className="font-medium text-slate-500">Med</span> = Claude MEDIUM (awaiting your ✓ in the review filter).
        The dashed line marks the rough ≥ 100-GOLD threshold for fine-tuning a per-species head.
      </p>
      <div className="flex gap-4 text-xs text-slate-600 mb-2">
        <span>
          <span className="font-semibold">{totals.training_ready_species}</span> species training-ready
          <span className="text-slate-400"> (≥ 100 gold)</span>
        </span>
        <span>
          <span className="font-semibold">{totals.review_queue_size}</span> in MEDIUM review queue
        </span>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(280, rows.length * 22)}>
        <BarChart data={rows} layout="vertical" margin={{ left: 10, right: 30, top: 4, bottom: 4 }}>
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, xMax]} tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="species" tick={{ fontSize: 11 }} width={140} />
          <Tooltip
            // Recharts passes the segment value as `v` and the original
            // row as `p.payload`; we show all three tiers + total so the
            // tooltip is one-stop for "how do I push this species over
            // the bar?"
            formatter={(v, _name, p) =>
              [`${v}`, `${_name}  (gold=${p.payload.gold}, high=${p.payload.high}, med=${p.payload.medium}, total=${p.payload.total})`]
            }
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {/* Stack order: gold first (left edge), then high, then med.
              Reading left → right matches descending trust. */}
          <Bar dataKey="gold" stackId="t" name="Gold" fill={TIER_COLOR_GOLD} />
          <Bar dataKey="high" stackId="t" name="High" fill={TIER_COLOR_HIGH} />
          <Bar dataKey="medium" stackId="t" name="Medium" fill={TIER_COLOR_MED} />
          {/* Threshold marker. Recharts doesn't have a great vertical
              line primitive on a vertical bar chart, but a zero-width
              Bar at x=100 styled as a dashed stroke approximates it. */}
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}

// ─── Top species table ───────────────────────────────────────────────────

function TopSpeciesTable({ totals }: { totals: StatsResponse["totals"] }) {
  if (totals.top_species.length === 0) return null;
  return (
    <section className="bg-white rounded-lg shadow-sm p-3">
      <h3 className="text-sm font-semibold mb-2">Top labeled species (all time)</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-slate-500 text-left">
            <th className="font-normal">Species</th>
            <th className="font-normal text-right">Labels</th>
            <th className="font-normal text-right">→ fine-tune</th>
          </tr>
        </thead>
        <tbody>
          {totals.top_species.map((s) => (
            <tr key={s.species} className="border-t border-slate-100">
              <td className="py-1">{s.species}</td>
              <td className="py-1 text-right tabular-nums">{s.count}</td>
              <td className="py-1 text-right text-xs text-slate-500">
                {s.count >= 50 ? "✓" : `${50 - s.count} more`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

// ─── Hour-of-day heatmap ─────────────────────────────────────────────────

function HourOfDayHeatmap({ daily }: { daily: DailyStats[] }) {
  // Sum each hour bucket across the displayed window. Detections are
  // captured by the camera in UTC (we run the Reolink in GMT+0 for
  // audio-correlation alignment), so the buckets are UTC hours.
  const counts = useMemo(() => {
    const out = new Array(24).fill(0);
    for (const d of daily) {
      const h = d.payload.hour_of_day;
      if (!h) continue;
      for (let i = 0; i < 24; i++) out[i] += h[i] ?? 0;
    }
    return out;
  }, [daily]);

  const max = Math.max(1, ...counts);

  return (
    <section className="bg-white rounded-lg shadow-sm p-3">
      <h3 className="text-sm font-semibold mb-2">Hour-of-day activity (UTC, 30d)</h3>
      <div className="grid grid-cols-12 gap-1">
        {counts.map((c, hour) => {
          const intensity = c / max;
          return (
            <div
              key={hour}
              title={`${hour.toString().padStart(2, "0")}:00 — ${c} detections`}
              className="aspect-square rounded-sm flex items-center justify-center text-[10px]"
              style={{
                backgroundColor: `rgba(47, 93, 61, ${0.08 + intensity * 0.85})`,
                color: intensity > 0.5 ? "white" : "#475569",
              }}
            >
              {hour}
            </div>
          );
        })}
      </div>
      <p className="text-xs text-slate-500 mt-1">
        Darker = more detections at that UTC hour, summed over the window.
      </p>
    </section>
  );
}

// ─── YOLO-confidence histogram (NAB vs species) ──────────────────────────

function YoloConfidenceHist({ daily }: { daily: DailyStats[] }) {
  const buckets = useMemo(() => {
    const nab = new Array(10).fill(0);
    const species = new Array(10).fill(0);
    for (const d of daily) {
      const h = d.payload.yolo_confidence_hist;
      if (!h) continue;
      for (let i = 0; i < 10; i++) {
        nab[i] += h.nab[i] ?? 0;
        species[i] += h.species[i] ?? 0;
      }
    }
    return Array.from({ length: 10 }, (_, i) => ({
      bucket: `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`,
      "Not a bird": nab[i],
      "Real species": species[i],
    }));
  }, [daily]);

  const total = buckets.reduce((s, b) => s + b["Not a bird"] + b["Real species"], 0);
  if (total === 0) {
    return (
      <section className="bg-white rounded-lg shadow-sm p-3">
        <h3 className="text-sm font-semibold mb-2">YOLO confidence: NAB vs species (30d)</h3>
        <p className="text-xs text-slate-500">No labeled detections with YOLO confidence in the window.</p>
      </section>
    );
  }

  return (
    <section className="bg-white rounded-lg shadow-sm p-3">
      <h3 className="text-sm font-semibold mb-2">YOLO confidence: NAB vs species (30d)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={buckets} margin={{ left: -10, right: 10, top: 4, bottom: 4 }}>
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
          <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="Not a bird" fill={COLOR_RED} />
          <Bar dataKey="Real species" fill={COLOR_FOREST} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-slate-500 mt-1">
        If high-confidence buckets are still NAB-heavy, raising
        BIRD_CONFIDENCE_THRESHOLD wouldn't help.
      </p>
    </section>
  );
}

// ─── Location heatmaps ──────────────────────────────────────────────────

/**
 * Three PNGs rendered server-side by `scripts/analyze_bird_locations.py`,
 * refreshed nightly at 02:20 UTC by the worker cron (plus once on every
 * container start). Files live under `data/heatmaps/` which is exposed
 * via the existing `/media/` static mount — no API call needed.
 *
 * The `key` busts the PWA service worker's cache once a day so we don't
 * end up showing yesterday's image after the nightly re-render.
 */
function LocationHeatmaps() {
  const dayBust = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const tabs: { id: "density" | "small" | "size"; label: string; src: string; caption: string }[] = [
    {
      id: "density",
      label: "Density",
      src: `/media/heatmaps/location_heatmap.png?d=${dayBust}`,
      caption: "Where real-bird detections cluster. White contours mark the 25 / 50 / 75% density isolines. Red hatched cells are the current scene-mask hot zones — YOLO hits there get suppressed unless confidence beats the override. A density cluster sitting under hatching is a real bird at a NAB hotspot worth checking.",
    },
    {
      id: "small",
      label: "Small birds",
      src: `/media/heatmaps/small_birds_heatmap.png?d=${dayBust}`,
      caption: "Detections with bbox-diagonal below the median — the resolution-bottleneck zones. If these clusters sit far from the camera, a closer second camera (or zoom) there is the highest-leverage move.",
    },
    {
      id: "size",
      label: "Size grid",
      src: `/media/heatmaps/size_by_region.png?d=${dayBust}`,
      caption: "Median bbox diagonal per grid cell, color-coded. Cool = small birds; warm = big birds. The number in each cell is the sample count.",
    },
  ];
  const [active, setActive] = useState<typeof tabs[number]["id"]>("density");
  const tab = tabs.find((t) => t.id === active)!;

  return (
    <section className="bg-white rounded-lg shadow-sm p-3">
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <h3 className="text-sm font-semibold">Where the birds are</h3>
        <div className="flex gap-1 text-xs">
          {tabs.map((t) => (
            <button
              key={t.id}
              className={`px-2 py-1 rounded border ${
                t.id === active
                  ? "border-forest text-forest bg-cream font-medium"
                  : "border-slate-200 text-slate-600 hover:border-forest hover:text-forest"
              }`}
              onClick={() => setActive(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>
      <img
        src={tab.src}
        alt={tab.label}
        className="w-full rounded border border-slate-100"
        loading="lazy"
        onError={(e) => {
          // Show a small placeholder if the cron hasn't rendered yet
          // (fresh container, missing background frame, etc.).
          (e.currentTarget as HTMLImageElement).style.display = "none";
        }}
      />
      <p className="text-xs text-slate-500 mt-1">{tab.caption}</p>
    </section>
  );
}


// ─── Page ────────────────────────────────────────────────────────────────

export default function Stats() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["stats", 30],
    queryFn: () => fetchStats(30),
    refetchInterval: 5 * 60 * 1000,  // 5 min — today's row is recomputed on read
  });

  if (isLoading) return <p className="text-slate-500">Loading stats…</p>;
  if (error || !data) {
    return (
      <p className="text-red-600 text-sm">
        Couldn't load stats: {(error as Error)?.message ?? "unknown error"}
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <HeadlineCards data={data} />
      <FunnelChart daily={data.daily} />
      <RatesChart daily={data.daily} />
      <TrainingDataCard totals={data.totals} />
      <SpeciesAccuracy totals={data.totals} />
      <TopSpeciesTable totals={data.totals} />
      <LocationHeatmaps />
      <div className="grid gap-4 md:grid-cols-2">
        <HourOfDayHeatmap daily={data.daily} />
        <YoloConfidenceHist daily={data.daily} />
      </div>
      <p className="text-xs text-slate-400 text-right">
        Updated {new Date(data.as_of).toLocaleString()}
      </p>
    </div>
  );
}
