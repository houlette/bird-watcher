import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
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
import { readTokens, tip, type Tokens } from "../lib/chartTheme";

// ─── Model-update markers ──────────────────────────────────────────────────
type ModelMarker = { date: string; label: string };
const MODEL_MARKERS: ModelMarker[] = [
  { date: "Jun 7", label: "NAB filter" },
  { date: "Jun 8", label: "birdclass-na" },
];

// Markers on adjacent days collide when both labels sit at the same
// height — stagger alternate labels down a line so each stays readable.
function markerLabel(m: ModelMarker, i: number, t: Tokens) {
  return { value: m.label, position: "top" as const, fontSize: 10, fill: t.slate, dy: (i % 2) * 12 };
}

// ─── Helpers ────────────────────────────────────────────────────────────────
function fmtPct(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${Math.round(x * 100)}%`;
}
function shortDate(iso: string): string {
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

// ─── Shared layout primitives ───────────────────────────────────────────────
function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`fg-card p-4 ${className}`}>{children}</section>;
}
function CardTitle({
  children,
  hint,
}: {
  children: React.ReactNode;
  hint?: React.ReactNode;
}) {
  return (
    <h3 className="font-serif text-[17px] font-medium text-ink mb-1">
      {children}
      {hint && <span className="ml-1.5 text-xs text-faint font-sans font-normal">{hint}</span>}
    </h3>
  );
}

// ─── Top-level cards ──────────────────────────────────────────────────────
function HeadlineCards({ data }: { data: StatsResponse }) {
  const today = data.daily[data.daily.length - 1];
  const yesterday = data.daily.length >= 2 ? data.daily[data.daily.length - 2] : null;
  const corrections30d = data.daily.reduce((s, d) => s + d.detections_user_corrected, 0);

  const cards: { label: string; value: string; sub?: string }[] = [
    { label: "Clips today", value: String(today.clips_received), sub: yesterday ? `yesterday ${yesterday.clips_received}` : undefined },
    { label: "Detections today", value: String(today.detections_total), sub: yesterday ? `yesterday ${yesterday.detections_total}` : undefined },
    { label: "Corrections (30d)", value: String(corrections30d), sub: `${data.totals.corrections_total} all-time` },
    { label: "Label backlog", value: String(data.totals.pending_backlog), sub: "model-labeled, awaiting you" },
    { label: "Ready to fine-tune", value: String(data.totals.ready_to_fine_tune_species), sub: "species ≥ 50 labels" },
    { label: "Mask-suppressed today", value: String(today.detections_scene_mask_suppressed ?? 0), sub: "YOLO hits dropped by hot-zone mask" },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      {cards.map((c) => (
        <div key={c.label} className="fg-card p-3.5">
          <div className="fg-overline">{c.label}</div>
          <div className="font-serif text-3xl text-ink leading-none mt-1.5 tnum">{c.value}</div>
          {c.sub && <div className="text-[11px] text-faint mt-1.5">{c.sub}</div>}
        </div>
      ))}
    </div>
  );
}

// ─── Funnel chart ─────────────────────────────────────────────────────────
function FunnelChart({ daily }: { daily: DailyStats[] }) {
  const t = readTokens();
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
    <Card>
      <CardTitle>Pipeline funnel (30d)</CardTitle>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows} margin={{ left: -10, right: 10, top: 8, bottom: 4 }}>
          <CartesianGrid stroke={t.grid} strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: t.axis }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 11, fill: t.axis }} />
          <Tooltip {...tip(t)} />
          <Legend wrapperStyle={{ fontSize: 12, color: t.ink }} />
          {MODEL_MARKERS.map((m, i) => (
            <ReferenceLine key={m.date} x={m.date} stroke={t.slate} strokeDasharray="2 3"
              label={markerLabel(m, i, t)} />
          ))}
          <Line type="monotone" dataKey="Clips" stroke={t.slate} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Detections" stroke={t.leaf} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Classifier-labeled" stroke={t.blue} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Corrected" stroke={t.sand} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Mask-suppressed" stroke={t.rust} strokeWidth={2} strokeDasharray="4 3" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
}

// ─── Rate chart ────────────────────────────────────────────────────────────
function RatesChart({ daily }: { daily: DailyStats[] }) {
  const t = readTokens();
  const rows = useMemo(
    () =>
      daily.map((d) => ({
        date: shortDate(d.date),
        "YOLO bird rate": d.yolo_bird_rate,
        "Classifier label rate": d.classifier_label_rate,
        "NAB share of corrections": d.user_fp_rate,
        "Top-1 hit rate (on reviewed)": d.classifier_accuracy,
      })),
    [daily],
  );

  return (
    <Card>
      <CardTitle>Pipeline rates (30d)</CardTitle>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows} margin={{ left: -10, right: 10, top: 8, bottom: 4 }}>
          <CartesianGrid stroke={t.grid} strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: t.axis }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 11, fill: t.axis }} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <Tooltip {...tip(t)} formatter={(v) => (v == null ? "—" : fmtPct(Number(v)))} />
          <Legend wrapperStyle={{ fontSize: 12, color: t.ink }} />
          {MODEL_MARKERS.map((m, i) => (
            <ReferenceLine key={m.date} x={m.date} stroke={t.slate} strokeDasharray="2 3"
              label={markerLabel(m, i, t)} />
          ))}
          <Line type="monotone" dataKey="YOLO bird rate" stroke={t.leaf} strokeWidth={2} dot={false} connectNulls />
          <Line type="monotone" dataKey="Classifier label rate" stroke={t.blue} strokeWidth={2} dot={false} connectNulls />
          <Line type="monotone" dataKey="NAB share of corrections" stroke={t.rust} strokeWidth={2} dot={false} connectNulls />
          <Line type="monotone" dataKey="Top-1 hit rate (on reviewed)" stroke={t.sand} strokeWidth={2} dot={false} connectNulls />
        </LineChart>
      </ResponsiveContainer>
      <div className="text-xs text-muted mt-2 space-y-1">
        <p><span className="font-semibold" style={{ color: t.leaf }}>YOLO bird rate</span> — daylight clips where YOLO found anything (clips with detections ÷ daylight clips).</p>
        <p><span className="font-semibold" style={{ color: t.blue }}>Classifier label rate</span> — detections the species classifier was confident enough to label (vs. "Unidentified"). Denominator: all detections that day.</p>
        <p><span className="font-semibold" style={{ color: t.rust }}>NAB share of corrections</span> — of the detections you reviewed today, the fraction you marked "Not a bird." Biased by what you chose to review.</p>
        <p><span className="font-semibold" style={{ color: t.sand }}>Top-1 hit rate (on reviewed)</span> — of detections where you confirmed a real species, the fraction where the classifier's top-1 already matched. Biased downward.</p>
      </div>
    </Card>
  );
}

// ─── Per-species accuracy ─────────────────────────────────────────────────
function SpeciesAccuracy({ totals }: { totals: StatsResponse["totals"] }) {
  const t = readTokens();
  const rows = totals.species_accuracy.map((s) => ({ species: s.species, accuracy: s.accuracy, n: s.n }));
  if (rows.length === 0) {
    return (
      <Card>
        <CardTitle>Per-species accuracy</CardTitle>
        <p className="text-xs text-muted">No species has ≥ 5 ground-truth labels yet.</p>
      </Card>
    );
  }
  return (
    <Card>
      <CardTitle hint="(top 15 by label count)">Classifier vs corrections</CardTitle>
      <p className="text-xs text-muted mb-2">
        Of the times you labeled a detection as species X, how often the classifier had already
        guessed X (top-1). Biased down — you mostly correct mistakes, so low bars mean "you only
        see it when it's wrong," not "hopeless."
      </p>
      <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 24)}>
        <BarChart data={rows} layout="vertical" margin={{ left: 10, right: 30, top: 4, bottom: 4 }}>
          <CartesianGrid stroke={t.grid} strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: t.axis }} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          {/* interval={0}: render EVERY species label. Recharts' auto
              interval skips alternate category ticks when they're tight,
              which left bars floating with no species name next to them. */}
          <YAxis type="category" dataKey="species" tick={{ fontSize: 11, fill: t.axis }} width={140} interval={0} />
          <Tooltip {...tip(t)} formatter={(v, _n, p) => [`${fmtPct(Number(v))} of ${p.payload.n}`, "Top-1 accuracy"]} />
          <Bar dataKey="accuracy" fill={t.leaf} radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}

// ─── Training-data progress ────────────────────────────────────────────────
function TrainingDataCard({ totals }: { totals: StatsResponse["totals"] }) {
  const t = readTokens();
  const rows = totals.training_data ?? [];
  if (rows.length === 0) {
    return (
      <Card>
        <CardTitle>Training data progress</CardTitle>
        <p className="text-xs text-muted">No labeled species yet.</p>
      </Card>
    );
  }
  const maxTotal = Math.max(...rows.map((r) => r.total));
  const xMax = Math.ceil((maxTotal * 1.05) / 100) * 100;
  const GOLD = t.leaf;
  const HIGH = t.sand;
  // slate, not grid: the gridline color is intentionally near-invisible
  // against the card background, which made the Medium segments (and the
  // legend swatch) unreadable, especially in dark mode.
  const MED = t.slate;

  return (
    <Card>
      <CardTitle hint="(top 25 by label count)">Training data progress</CardTitle>
      <p className="text-xs text-muted mb-2">
        Per-species labeled detections, stacked by trust tier.{" "}
        <span className="font-semibold" style={{ color: GOLD }}>Gold</span> = you verified;{" "}
        <span className="font-semibold" style={{ color: HIGH }}>High</span> = Claude HIGH (auto-committed, unreviewed);{" "}
        <span className="font-semibold" style={{ color: MED }}>Med</span> = Claude MEDIUM (awaiting your ✓ in the review filter).
      </p>
      <div className="flex gap-4 text-xs text-muted mb-2">
        <span><span className="font-semibold text-ink tnum">{totals.training_ready_species}</span> training-ready <span className="text-faint">(≥ 100 gold)</span></span>
        <span><span className="font-semibold text-ink tnum">{totals.review_queue_size}</span> in MEDIUM review queue</span>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(280, rows.length * 22)}>
        <BarChart data={rows} layout="vertical" margin={{ left: 10, right: 30, top: 4, bottom: 4 }}>
          <CartesianGrid stroke={t.grid} strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, xMax]} tick={{ fontSize: 11, fill: t.axis }} />
          <YAxis type="category" dataKey="species" tick={{ fontSize: 11, fill: t.axis }} width={140} interval={0} />
          <Tooltip {...tip(t)}
            formatter={(v, _n, p) => [`${v}`, `${_n}  (gold=${p.payload.gold}, high=${p.payload.high}, med=${p.payload.medium}, total=${p.payload.total})`]} />
          <Legend wrapperStyle={{ fontSize: 11, color: t.ink }} />
          <Bar dataKey="gold" stackId="t" name="Gold" fill={GOLD} />
          <Bar dataKey="high" stackId="t" name="High" fill={HIGH} />
          <Bar dataKey="medium" stackId="t" name="Medium" fill={MED} />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}

// ─── Top species table ─────────────────────────────────────────────────────
function TopSpeciesTable({ totals }: { totals: StatsResponse["totals"] }) {
  if (totals.top_species.length === 0) return null;
  return (
    <Card>
      <CardTitle>Top labeled species (all time)</CardTitle>
      <table className="w-full text-sm">
        <thead>
          <tr className="fg-overline text-left">
            <th className="font-semibold pb-1">Species</th>
            <th className="font-semibold text-right pb-1">Labels</th>
            <th className="font-semibold text-right pb-1">→ fine-tune</th>
          </tr>
        </thead>
        <tbody>
          {totals.top_species.map((s) => (
            <tr key={s.species} className="border-t border-line">
              <td className="py-1.5 text-ink">{s.species}</td>
              <td className="py-1.5 text-right tnum text-ink">{s.count}</td>
              <td className="py-1.5 text-right text-xs text-muted tnum">
                {s.count >= 50 ? <span className="text-leaf font-semibold">✓</span> : `${50 - s.count} more`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

// ─── Image-quality time series ─────────────────────────────────────────────
type QualityMetric = "sharpness" | "brightness" | "area_px";
type QualityRow = { date: string; p25: number | null; p50: number | null; p75: number | null; n: number };

function ImageQualityPanel({
  daily, metric, title, unit, badThreshold, badLabel,
}: {
  daily: DailyStats[];
  metric: QualityMetric;
  title: string;
  unit: string;
  badThreshold?: number;
  badLabel?: string;
}) {
  const t = readTokens();
  const rows = useMemo<QualityRow[]>(
    () =>
      daily.map((d) => {
        const cq = d.payload.crop_quality?.[metric];
        return { date: shortDate(d.date), p25: cq?.p25 ?? null, p50: cq?.p50 ?? null, p75: cq?.p75 ?? null, n: cq?.n ?? 0 };
      }),
    [daily, metric],
  );

  const hasData = rows.some((r) => r.p50 !== null);
  if (!hasData) {
    return (
      <div className="fg-card p-3.5">
        <h4 className="text-xs font-semibold text-ink mb-1">{title}</h4>
        <p className="text-xs text-muted">
          No data yet — populated as new detections come in, or backfill via{" "}
          <code className="bg-panel border border-line px-1 rounded">scripts/backfill_crop_quality.py</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="fg-card p-3.5">
      <h4 className="text-xs font-semibold text-ink mb-1">{title}</h4>
      <ResponsiveContainer width="100%" height={150}>
        <ComposedChart data={rows} margin={{ left: -10, right: 10, top: 4, bottom: 4 }}>
          <CartesianGrid stroke={t.grid} strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: t.axis }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 10, fill: t.axis }} />
          <Tooltip {...tip(t)}
            formatter={(v, name, p) => {
              if (v == null) return "—";
              const n = p.payload?.n ?? 0;
              const valStr = `${Number(v).toFixed(metric === "area_px" ? 0 : 1)} ${unit}`;
              return [valStr, `${name}  (n=${n})`];
            }} />
          {MODEL_MARKERS.map((m) => (
            <ReferenceLine key={m.date} x={m.date} stroke={t.slate} strokeDasharray="2 3" />
          ))}
          {badThreshold !== undefined && (
            <ReferenceLine y={badThreshold} stroke={t.rust} strokeDasharray="3 3"
              label={{ value: badLabel ?? `< ${badThreshold}`, position: "insideBottomRight", fontSize: 10, fill: t.rust }} />
          )}
          <Area type="monotone" dataKey="p25" stackId="band" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area type="monotone"
            dataKey={(r: QualityRow) => (r.p25 !== null && r.p75 !== null ? r.p75 - r.p25 : null)}
            name="p25–p75" stackId="band" stroke="none" fill={t.band} isAnimationActive={false} />
          <Line type="monotone" dataKey="p50" name="median" stroke={t.teal} strokeWidth={2} dot={false} connectNulls />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function ImageQuality({ daily }: { daily: DailyStats[] }) {
  return (
    <Card>
      <CardTitle>Image quality (30d)</CardTitle>
      <p className="text-xs text-muted mb-3">
        Per-day crop-quality percentiles (teal line = median, shaded band = p25–p75). Computed
        after CLAHE, before classifier resize. Sharpness is the bottleneck — the species
        classifier maxes out at whatever this line allows.
      </p>
      <div className="grid gap-3 md:grid-cols-3">
        <ImageQualityPanel daily={daily} metric="sharpness" title="Sharpness (Laplacian variance)" unit="" badThreshold={30} badLabel="~too blurry" />
        <ImageQualityPanel daily={daily} metric="brightness" title="Brightness (mean grayscale, 0–255)" unit="" badThreshold={30} badLabel="~too dark" />
        <ImageQualityPanel daily={daily} metric="area_px" title="Crop area (px²)" unit="px²" />
      </div>
    </Card>
  );
}

// ─── Hour-of-day heatmap ───────────────────────────────────────────────────
function HourOfDayHeatmap({ daily }: { daily: DailyStats[] }) {
  const counts = useMemo(() => {
    // Payload buckets are UTC hours; rotate into the viewer's local zone
    // so "birds at 7am" means 7am at the feeder, not 7am in Greenwich.
    // Current offset only (DST shifts the window by an hour at the
    // edges) — fine for a glanceable activity grid.
    const utc = new Array(24).fill(0);
    for (const d of daily) {
      const h = d.payload.hour_of_day;
      if (!h) continue;
      for (let i = 0; i < 24; i++) utc[i] += h[i] ?? 0;
    }
    const offsetH = Math.round(-new Date().getTimezoneOffset() / 60);
    const local = new Array(24).fill(0);
    for (let u = 0; u < 24; u++) local[(u + offsetH + 24) % 24] = utc[u];
    return local;
  }, [daily]);
  const max = Math.max(1, ...counts);

  return (
    <Card>
      <CardTitle>Hour-of-day activity (local time, 30d)</CardTitle>
      <div className="grid grid-cols-12 gap-1">
        {counts.map((c, hour) => {
          const intensity = c / max;
          return (
            <div
              key={hour}
              title={`${hour.toString().padStart(2, "0")}:00 — ${c} detections`}
              className="aspect-square rounded-sm flex items-center justify-center text-[10px] tnum"
              style={{
                backgroundColor: `color-mix(in oklab, var(--accent) ${Math.round(10 + intensity * 80)}%, var(--panel))`,
                color: intensity > 0.5 ? "var(--card)" : "var(--muted)",
              }}
            >
              {hour}
            </div>
          );
        })}
      </div>
      <p className="text-xs text-muted mt-2">Darker = more detections at that local hour, summed over the window.</p>
    </Card>
  );
}

// ─── YOLO-confidence histogram ─────────────────────────────────────────────
function YoloConfidenceHist({ daily }: { daily: DailyStats[] }) {
  const t = readTokens();
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
      <Card>
        <CardTitle>YOLO confidence: NAB vs species (30d)</CardTitle>
        <p className="text-xs text-muted">No labeled detections with YOLO confidence in the window.</p>
      </Card>
    );
  }

  return (
    <Card>
      <CardTitle>YOLO confidence: NAB vs species (30d)</CardTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={buckets} margin={{ left: -10, right: 10, top: 4, bottom: 4 }}>
          <CartesianGrid stroke={t.grid} strokeDasharray="3 3" />
          <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: t.axis }} />
          <YAxis tick={{ fontSize: 11, fill: t.axis }} />
          <Tooltip {...tip(t)} />
          <Legend wrapperStyle={{ fontSize: 12, color: t.ink }} />
          <Bar dataKey="Not a bird" fill={t.rust} radius={[3, 3, 0, 0]} />
          <Bar dataKey="Real species" fill={t.leaf} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-muted mt-2">
        If high-confidence buckets are still NAB-heavy, raising BIRD_CONFIDENCE_THRESHOLD
        wouldn't help.
      </p>
    </Card>
  );
}

// ─── Location heatmaps ──────────────────────────────────────────────────────
function LocationHeatmaps() {
  const dayBust = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const tabs: { id: "density" | "small" | "size"; label: string; src: string; caption: string }[] = [
    { id: "density", label: "Density", src: `/media/heatmaps/location_heatmap.png?d=${dayBust}`,
      caption: "Where real-bird detections cluster. White contours mark the 25 / 50 / 75% density isolines. Red hatched cells are the current scene-mask hot zones — YOLO hits there get suppressed unless confidence beats the override." },
    { id: "small", label: "Small birds", src: `/media/heatmaps/small_birds_heatmap.png?d=${dayBust}`,
      caption: "Detections with bbox-diagonal below the median — the resolution-bottleneck zones. If these clusters sit far from the camera, a closer second camera (or zoom) there is the highest-leverage move." },
    { id: "size", label: "Size grid", src: `/media/heatmaps/size_by_region.png?d=${dayBust}`,
      caption: "Median bbox diagonal per grid cell, color-coded. Cool = small birds; warm = big birds. The number in each cell is the sample count." },
  ];
  const [active, setActive] = useState<typeof tabs[number]["id"]>("density");
  const tab = tabs.find((t) => t.id === active)!;

  return (
    <Card>
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <CardTitle>Where the birds are</CardTitle>
        <div className="flex gap-1.5 text-xs">
          {tabs.map((tb) => (
            <button
              key={tb.id}
              className={`rounded-full px-2.5 py-1 border transition-colors ${
                tb.id === active
                  ? "border-leaf text-surface"
                  : "border-line text-muted hover:border-leaf hover:text-leaf"
              }`}
              style={tb.id === active ? { background: "var(--accent)" } : undefined}
              onClick={() => setActive(tb.id)}
            >
              {tb.label}
            </button>
          ))}
        </div>
      </div>
      <img
        src={tab.src}
        alt={tab.label}
        className="w-full rounded-card border border-line"
        loading="lazy"
        onError={(e) => {
          (e.currentTarget as HTMLImageElement).style.display = "none";
        }}
      />
      <p className="text-xs text-muted mt-2">{tab.caption}</p>
    </Card>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────
export default function Stats() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["stats", 30],
    queryFn: () => fetchStats(30),
    refetchInterval: 5 * 60 * 1000,
  });

  if (isLoading) return <p className="text-muted mt-4">Loading stats…</p>;
  if (error || !data) {
    return (
      <p className="text-rust text-sm mt-4">
        Couldn't load stats: {(error as Error)?.message ?? "unknown error"}
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <div className="mb-1">
        <div className="fg-overline">Station diagnostics</div>
        <h2 className="font-serif font-medium text-2xl text-ink leading-tight mt-0.5">
          Pipeline & training stats
        </h2>
      </div>
      <HeadlineCards data={data} />
      <FunnelChart daily={data.daily} />
      <RatesChart daily={data.daily} />
      <ImageQuality daily={data.daily} />
      <TrainingDataCard totals={data.totals} />
      <SpeciesAccuracy totals={data.totals} />
      <TopSpeciesTable totals={data.totals} />
      <LocationHeatmaps />
      <div className="grid gap-4 md:grid-cols-2">
        <HourOfDayHeatmap daily={data.daily} />
        <YoloConfidenceHist daily={data.daily} />
      </div>
      <p className="text-xs text-faint text-right tnum">
        Updated {new Date(data.as_of).toLocaleString()}
      </p>
    </div>
  );
}
