import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";

import type { ArtFlight, ArtSun } from "../lib/api";

export type ArtMode = "flightlines" | "mandala" | "topography";

export interface ArtCanvasHandle {
  /** Render the current scene at print resolution and hand back a PNG blob. */
  exportPNG: (width?: number) => Promise<Blob | null>;
}

export interface ArtCanvasProps {
  flights: ArtFlight[];
  mode: ArtMode;
  sun: ArtSun;
  /**
   * Playback position as a fraction of the local day, or -1 for "show all".
   *
   * This is a ref, not a value, on purpose. Playback advances every frame,
   * and a number prop would re-render the parent and tear down this
   * component's animation loop sixty times a second. The loop reads the
   * ref directly and nothing above it re-renders.
   */
  progressRef: React.MutableRefObject<number>;
  hoveredId: number | null;
  pinnedId: number | null;
  onHover: (flight: ArtFlight | null) => void;
  onPick: (flight: ArtFlight | null) => void;
  /** Bumped by the parent when the theme flips, so the palette is re-read. */
  themeKey: string;
}

type RGB = [number, number, number];

interface Theme {
  isDark: boolean;
  paper: RGB;
  panel: RGB;
  surface: RGB;
  ink: RGB;
  muted: RGB;
  faint: RGB;
  line: RGB;
  leaf: RGB;
}

interface Prepared {
  flight: ArtFlight;
  /** Path resampled to a fixed count, still in normalised frame space. */
  pts: { x: number; y: number; t: number }[];
  /** Half-width per point, as a fraction of the canvas short edge. */
  half: number[];
  perch: { x: number; y: number };
  rgb: RGB;
  accent: RGB;
  /** Position in the stack of flights sharing this quarter-hour. */
  ring: number;
}

interface Field {
  cols: number;
  rows: number;
  /** Perch density, normalised to a 0-1 peak. */
  density: Float32Array;
  /** Mean plumage colour per cell, premultiplied by density. */
  tint: Float32Array;
  /** The field painted as an image, ready to scale up as an elevation wash. */
  wash: HTMLCanvasElement | null;
  /** One list of line segments per contour level, in normalised coords. */
  contours: number[][][];
}

interface Grain {
  x: number;
  y: number;
  tx: number;
  ty: number;
  size: number;
  tone: number;
  rgb: RGB;
}

const PATH_SAMPLES = 56;
const FIELD_COLS = 96;
const FIELD_ROWS = 54;
const GRAIN_COUNT = 7000;
const CONTOUR_LEVELS = [0.06, 0.13, 0.22, 0.34, 0.5, 0.7];
/**
 * How much of the day a flight takes to draw itself in during playback,
 * as a fraction. About eight minutes of feeder time, which at the slowest
 * speed is a leisurely stroke and at the fastest is a flick.
 */
const DRAW_WINDOW = 0.0055;

export const ArtCanvas = forwardRef<ArtCanvasHandle, ArtCanvasProps>(function ArtCanvas(
  props,
  ref
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  // Everything the render loop needs, mirrored so the loop itself is set up
  // exactly once. Synced from an effect rather than during render: a render
  // React later discards must not have written anything the loop can see.
  const liveRef = useRef(props);
  useEffect(() => {
    liveRef.current = props;
  });

  const themeRef = useRef<Theme>(FALLBACK_THEME);
  const fieldRef = useRef<Field | null>(null);
  const grainsRef = useRef<Grain[]>([]);
  const hoverRef = useRef<number | null>(null);

  // Resampling every flight is the expensive part and depends only on the
  // data, so it happens here rather than per frame.
  const prepared = useMemo(() => {
    const list = props.flights.map(prepare);
    assignRings(list);
    return list;
  }, [props.flights]);
  const preparedRef = useRef(prepared);
  useEffect(() => {
    preparedRef.current = prepared;
  }, [prepared]);

  // The perch-density field and the sand that settles onto it are derived
  // from the same data; rebuild both together when the day changes.
  useEffect(() => {
    fieldRef.current = buildField(prepared);
    grainsRef.current = scatterGrains(fieldRef.current, grainsRef.current);
  }, [prepared]);

  useEffect(() => {
    themeRef.current = readTheme();
  }, [props.themeKey]);

  useImperativeHandle(
    ref,
    () => ({
      exportPNG: (width = 3840) =>
        new Promise<Blob | null>((resolve) => {
          const height = Math.round((width * 9) / 16);
          const off = document.createElement("canvas");
          off.width = width;
          off.height = height;
          const ctx = off.getContext("2d");
          if (!ctx) return resolve(null);

          // Draw the scene as it stands, with the animated extras stilled:
          // an exported print shouldn't catch a pulse ring mid-breath.
          drawScene(ctx, width, height, {
            prepared: preparedRef.current,
            mode: liveRef.current.mode,
            sun: liveRef.current.sun,
            progress: liveRef.current.progressRef.current,
            hoveredId: null,
            pinnedId: liveRef.current.pinnedId,
            theme: themeRef.current,
            field: fieldRef.current,
            grains: grainsRef.current,
            settled: true,
          });
          off.toBlob((b) => resolve(b), "image/png");
        }),
    }),
    []
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const box = boxRef.current;
    if (!canvas || !box) return;

    let raf = 0;
    let cssW = 0;
    let cssH = 0;

    const resize = () => {
      const rect = box.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      cssW = Math.max(1, rect.width);
      cssH = Math.max(1, (rect.width * 9) / 16);
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      canvas.style.height = `${cssH}px`;
    };

    const ro = new ResizeObserver(resize);
    ro.observe(box);
    resize();

    // ── Hit testing ──────────────────────────────────────────────────
    const hitTest = (clientX: number, clientY: number): ArtFlight | null => {
      const rect = canvas.getBoundingClientRect();
      const mx = (clientX - rect.left) / rect.width;
      const my = (clientY - rect.top) / rect.height;
      const { mode, progressRef } = liveRef.current;
      const progress = progressRef.current;

      let best: ArtFlight | null = null;
      let bestDist = 0.035;

      for (const p of preparedRef.current) {
        if (revealFraction(p.flight, progress) <= 0) continue;
        if (mode === "mandala") {
          // In polar projection the drawn node is nowhere near the path,
          // so test against the node itself.
          const node = mandalaNode(p, cssW, cssH);
          const d = Math.hypot(node.x / cssW - mx, node.y / cssH - my);
          if (d < bestDist) {
            bestDist = d;
            best = p.flight;
          }
          continue;
        }
        const pts = mode === "topography" ? [p.perch] : p.pts;
        for (const pt of pts) {
          const d = Math.hypot(pt.x - mx, pt.y - my);
          if (d < bestDist) {
            bestDist = d;
            best = p.flight;
          }
        }
      }
      return best;
    };

    const onMove = (e: PointerEvent) => {
      const hit = hitTest(e.clientX, e.clientY);
      const id = hit?.detection_id ?? null;
      if (id !== hoverRef.current) {
        hoverRef.current = id;
        liveRef.current.onHover(hit);
      }
    };
    const onLeave = () => {
      if (hoverRef.current !== null) {
        hoverRef.current = null;
        liveRef.current.onHover(null);
      }
    };
    const onDown = (e: PointerEvent) => {
      // Touch never fires a hover first, so resolve the hit on the tap.
      liveRef.current.onPick(hitTest(e.clientX, e.clientY));
    };

    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("pointerdown", onDown);

    // ── Frame loop ───────────────────────────────────────────────────
    const frame = () => {
      const ctx = canvas.getContext("2d");
      if (ctx && cssW > 0) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawScene(ctx, cssW, cssH, {
          prepared: preparedRef.current,
          mode: liveRef.current.mode,
          sun: liveRef.current.sun,
          progress: liveRef.current.progressRef.current,
          hoveredId: liveRef.current.hoveredId,
          pinnedId: liveRef.current.pinnedId,
          theme: themeRef.current,
          field: fieldRef.current,
          grains: grainsRef.current,
          settled: false,
        });
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("pointerdown", onDown);
    };
  }, []);

  return (
    <div
      ref={boxRef}
      className="relative w-full overflow-hidden rounded-card border border-line bg-surface select-none touch-manipulation"
      style={{ aspectRatio: "16 / 9" }}
    >
      <canvas ref={canvasRef} className="block w-full cursor-crosshair" />
    </div>
  );
});

// ── Scene ───────────────────────────────────────────────────────────────

interface Scene {
  prepared: Prepared[];
  mode: ArtMode;
  sun: ArtSun;
  progress: number;
  hoveredId: number | null;
  pinnedId: number | null;
  theme: Theme;
  field: Field | null;
  grains: Grain[];
  /** True for exports: settle the sand where it belongs instead of easing. */
  settled: boolean;
}

function drawScene(ctx: CanvasRenderingContext2D, w: number, h: number, s: Scene) {
  paintGround(ctx, w, h, s.theme);
  if (s.mode === "flightlines") drawFlightlines(ctx, w, h, s);
  else if (s.mode === "mandala") drawMandala(ctx, w, h, s);
  else drawTopography(ctx, w, h, s);
}

function paintGround(ctx: CanvasRenderingContext2D, w: number, h: number, t: Theme) {
  // A shallow vignette between two theme tokens: enough depth that strokes
  // sit in a space, not so much that the canvas stops looking like the app.
  const g = ctx.createRadialGradient(w * 0.5, h * 0.46, w * 0.05, w * 0.5, h * 0.5, w * 0.78);
  g.addColorStop(0, rgb(t.surface));
  g.addColorStop(1, rgb(t.isDark ? t.paper : t.panel));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
}

/**
 * How much of a flight is drawn at the current playback position.
 *
 * Returns 0 before the bird arrives, ramps to 1 across DRAW_WINDOW so the
 * stroke paints itself on, and stays at 1 afterwards — the day accumulates
 * rather than scrolling past.
 */
function revealFraction(f: ArtFlight, progress: number): number {
  if (progress < 0) return 1;
  if (f.time_of_day > progress) return 0;
  return Math.min(1, (progress - f.time_of_day) / DRAW_WINDOW);
}

// ── Mode 1: Flightlines ─────────────────────────────────────────────────

function drawFlightlines(ctx: CanvasRenderingContext2D, w: number, h: number, s: Scene) {
  const { theme } = s;
  const U = Math.min(w, h);

  drawFeederGhost(ctx, w, h, theme);

  // Ink glazes over paper, light adds onto dark. Both let overlapping
  // flights build density where the yard is busiest, which is the point.
  ctx.save();
  ctx.globalCompositeOperation = theme.isDark ? "screen" : "multiply";

  for (const p of s.prepared) {
    const reveal = revealFraction(p.flight, s.progress);
    if (reveal <= 0) continue;

    const focus = p.flight.detection_id === s.hoveredId || p.flight.detection_id === s.pinnedId;
    const dim = (s.hoveredId !== null || s.pinnedId !== null) && !focus;
    const alpha = (theme.isDark ? 0.4 : 0.34) * (focus ? 1.5 : dim ? 0.28 : 1);

    strokeRibbon(ctx, p, w, h, U, reveal, p.rgb, alpha);
  }
  ctx.restore();

  // Perch blooms and the travelling bird sit above the ink, in normal
  // compositing, so they stay legible however dense the ribbons get.
  for (const p of s.prepared) {
    const reveal = revealFraction(p.flight, s.progress);
    if (reveal <= 0) continue;
    const focus = p.flight.detection_id === s.hoveredId || p.flight.detection_id === s.pinnedId;
    const dim = (s.hoveredId !== null || s.pinnedId !== null) && !focus;
    if (dim && !focus) continue;

    const px = p.perch.x * w;
    const py = p.perch.y * h;
    const r = U * 0.009 * massScale(p.flight.style.mass_g) * (focus ? 2.4 : 1);

    const sprite = bloomSprite(p.rgb);
    if (sprite) {
      ctx.globalAlpha = focus ? 0.55 : 0.17;
      ctx.drawImage(sprite, px - r, py - r, r * 2, r * 2);
      ctx.globalAlpha = 1;
    }

    // The bird's contrasting plumage note — a mask, an epaulet, a belly —
    // as a small core inside the bloom.
    ctx.fillStyle = rgba(p.accent, focus ? 0.85 : 0.4);
    ctx.beginPath();
    ctx.arc(px, py, Math.max(0.8, r * 0.16), 0, Math.PI * 2);
    ctx.fill();

    if (p.flight.audio_confirmed) {
      // Haikubox heard it too. A steady ring, not a pulse: the fact is
      // static, and a canvas full of throbbing circles is a headache.
      ctx.strokeStyle = rgba(theme.leaf, focus ? 0.9 : 0.5);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(px, py, r * 1.35, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (reveal < 1) {
      // The bird itself, still on its way in.
      const at = p.pts[Math.min(p.pts.length - 1, Math.floor(reveal * (p.pts.length - 1)))];
      ctx.fillStyle = rgba(p.rgb, 0.95);
      ctx.beginPath();
      ctx.arc(at.x * w, at.y * h, U * 0.004, 0, Math.PI * 2);
      ctx.fill();
    }

    if (focus) {
      ctx.strokeStyle = rgba(theme.ink, 0.75);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(px, py, r * 1.7, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

/**
 * Stroke one flight as a variable-width ribbon.
 *
 * Canvas can't taper a stroke, so the ribbon is a filled polygon: out along
 * one side of the path, back along the other. Width comes from body mass,
 * tapers to nothing at both ends, and swells where the bird slows down —
 * which puts the heaviest ink on the perch, the way a loaded brush does.
 */
function strokeRibbon(
  ctx: CanvasRenderingContext2D,
  p: Prepared,
  w: number,
  h: number,
  U: number,
  reveal: number,
  color: RGB,
  alpha: number
) {
  const n = Math.max(2, Math.floor(p.pts.length * reveal));
  if (n < 2) return;

  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const { nx, ny } = normalAt(p.pts, i, w, h);
    const hw = p.half[i] * U;
    ctx.lineTo(p.pts[i].x * w + nx * hw, p.pts[i].y * h + ny * hw);
  }
  for (let i = n - 1; i >= 0; i--) {
    const { nx, ny } = normalAt(p.pts, i, w, h);
    const hw = p.half[i] * U;
    ctx.lineTo(p.pts[i].x * w - nx * hw, p.pts[i].y * h - ny * hw);
  }
  ctx.closePath();
  ctx.fillStyle = rgba(color, alpha);
  ctx.fill();
}

/** 1 on a straight run, falling toward 0.15 as the path doubles back. */
function bendLimit(pts: { x: number; y: number }[], i: number): number {
  if (i === 0 || i === pts.length - 1) return 1;
  const ax = pts[i].x - pts[i - 1].x;
  const ay = pts[i].y - pts[i - 1].y;
  const bx = pts[i + 1].x - pts[i].x;
  const by = pts[i + 1].y - pts[i].y;
  const la = Math.hypot(ax, ay) || 1e-6;
  const lb = Math.hypot(bx, by) || 1e-6;
  const cos = (ax * bx + ay * by) / (la * lb);
  return Math.min(1, Math.max(0.15, (cos + 1) * 0.7));
}

function normalAt(pts: { x: number; y: number; t: number }[], i: number, w: number, h: number) {
  const a = pts[Math.max(0, i - 1)];
  const b = pts[Math.min(pts.length - 1, i + 1)];
  const tx = (b.x - a.x) * w;
  const ty = (b.y - a.y) * h;
  const len = Math.hypot(tx, ty) || 1e-4;
  return { nx: -ty / len, ny: tx / len };
}

/** The feeder's own outline, ghosted in, so the flights have somewhere to be. */
function drawFeederGhost(ctx: CanvasRenderingContext2D, w: number, h: number, t: Theme) {
  ctx.save();
  ctx.strokeStyle = rgba(t.line, t.isDark ? 0.55 : 0.9);
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 7]);
  ctx.beginPath();
  ctx.moveTo(w * 0.62, 0);
  ctx.lineTo(w * 0.62, h * 0.5);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(w * 0.62, h * 0.55, w * 0.13, h * 0.06, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

// ── Mode 2: Celestial mandala ───────────────────────────────────────────

function mandalaGeom(w: number, h: number) {
  return { cx: w / 2, cy: h / 2, maxR: Math.min(w, h) * 0.4, step: Math.min(w, h) * 0.026 };
}

/** Quarter-hour slices. Fine enough to separate visits, coarse enough to stack. */
const MANDALA_BINS = 96;

/**
 * Stack flights that share a slice of the day outward from the hub.
 *
 * The obvious radial encoding — how high in frame the bird perched — turns
 * out to carry almost nothing: they all use the same feeder, so every node
 * lands on the same thin annulus and the dial's whole radial dimension goes
 * to waste. Stacking instead makes the radius mean traffic, so a busy dawn
 * grows a long spoke and a quiet afternoon a stub, and the dial finally
 * says something you can read across the room.
 */
function assignRings(list: Prepared[]) {
  const used = new Map<number, number>();
  for (const p of [...list].sort((a, b) => a.flight.time_of_day - b.flight.time_of_day)) {
    const bin = Math.floor(p.flight.time_of_day * MANDALA_BINS);
    const k = used.get(bin) ?? 0;
    used.set(bin, k + 1);
    p.ring = k;
  }
}

/** Where one flight's bead sits on the dial: angle is the hour, radius the stack. */
function mandalaNode(p: Prepared, w: number, h: number) {
  const { cx, cy, maxR, step } = mandalaGeom(w, h);
  const angle = p.flight.time_of_day * Math.PI * 2 - Math.PI / 2;
  const r = Math.min(maxR * 0.95, maxR * 0.17 + p.ring * step);
  return { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r, angle, r, step };
}

function drawMandala(ctx: CanvasRenderingContext2D, w: number, h: number, s: Scene) {
  const { theme, sun } = s;
  const { cx, cy, maxR, step } = mandalaGeom(w, h);
  const U = Math.min(w, h);
  const toAngle = (day: number) => day * Math.PI * 2 - Math.PI / 2;

  // Night: the hours the camera slept, as a band on the rim rather than a
  // pie slice. A wedge reaching the hub would imply something about the
  // radial axis, and the radial axis here counts birds, not darkness.
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, maxR, toAngle(sun.sunset), toAngle(sun.sunrise));
  ctx.arc(cx, cy, maxR * 0.84, toAngle(sun.sunrise), toAngle(sun.sunset), true);
  ctx.closePath();
  ctx.fillStyle = rgba(theme.ink, theme.isDark ? 0.34 : 0.1);
  ctx.fill();
  ctx.restore();

  // Hour ring and ticks.
  ctx.strokeStyle = rgba(theme.line, 1);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, maxR, 0, Math.PI * 2);
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let hour = 0; hour < 24; hour++) {
    const a = (hour / 24) * Math.PI * 2 - Math.PI / 2;
    const major = hour % 6 === 0;
    const len = major ? U * 0.03 : hour % 3 === 0 ? U * 0.019 : U * 0.01;
    ctx.strokeStyle = rgba(major ? theme.muted : theme.line, 1);
    ctx.lineWidth = major ? 1.2 : 0.8;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * (maxR - len), cy + Math.sin(a) * (maxR - len));
    ctx.lineTo(cx + Math.cos(a) * maxR, cy + Math.sin(a) * maxR);
    ctx.stroke();

    if (major) {
      ctx.font = `600 ${Math.round(U * 0.026)}px "Hanken Grotesk", system-ui, sans-serif`;
      ctx.fillStyle = rgba(theme.faint, 1);
      const label = hour === 0 ? "12a" : hour === 12 ? "noon" : hour < 12 ? `${hour}a` : `${hour - 12}p`;
      ctx.fillText(label, cx + Math.cos(a) * (maxR + U * 0.05), cy + Math.sin(a) * (maxR + U * 0.05));
    }
  }

  // Count rings, so the length of a spoke can be read as a number.
  ctx.strokeStyle = rgba(theme.line, 0.9);
  ctx.lineWidth = 0.7;
  for (let k = 5; k * step + maxR * 0.17 < maxR; k += 5) {
    ctx.beginPath();
    ctx.arc(cx, cy, maxR * 0.17 + k * step, 0, Math.PI * 2);
    ctx.stroke();
  }

  const visible = s.prepared.filter((p) => revealFraction(p.flight, s.progress) > 0);

  // Chords joining successive sightings of one species: that bird's day,
  // drawn as a constellation across the dial.
  const bySpecies = new Map<string, Prepared[]>();
  for (const p of visible) {
    const list = bySpecies.get(p.flight.species);
    if (list) list.push(p);
    else bySpecies.set(p.flight.species, [p]);
  }
  ctx.save();
  ctx.lineWidth = 0.8;
  for (const group of bySpecies.values()) {
    if (group.length < 2) continue;
    ctx.strokeStyle = rgba(group[0].rgb, theme.isDark ? 0.26 : 0.24);
    ctx.beginPath();
    for (let i = 0; i < group.length; i++) {
      const node = mandalaNode(group[i], w, h);
      if (i === 0) {
        ctx.moveTo(node.x, node.y);
      } else {
        // Bow each chord toward the hub so overlapping species stay apart.
        const prev = mandalaNode(group[i - 1], w, h);
        ctx.quadraticCurveTo(
          (prev.x + node.x) / 2 * 0.68 + cx * 0.32,
          (prev.y + node.y) / 2 * 0.68 + cy * 0.32,
          node.x,
          node.y
        );
      }
    }
    ctx.stroke();
  }
  ctx.restore();

  // Beads. Each one's stem reaches back toward the bead below it, so a
  // stack renders as one spoke banded by species.
  for (const p of visible) {
    const node = mandalaNode(p, w, h);
    const focus = p.flight.detection_id === s.hoveredId || p.flight.detection_id === s.pinnedId;
    const dim = (s.hoveredId !== null || s.pinnedId !== null) && !focus;
    const r = U * 0.0055 * massScale(p.flight.style.mass_g) * (focus ? 2.2 : 1);

    ctx.globalAlpha = dim ? 0.28 : 1;

    ctx.strokeStyle = rgba(p.rgb, 0.6);
    ctx.lineWidth = focus ? 3 : 2;
    ctx.beginPath();
    ctx.moveTo(
      cx + Math.cos(node.angle) * Math.max(maxR * 0.11, node.r - step),
      cy + Math.sin(node.angle) * Math.max(maxR * 0.11, node.r - step)
    );
    ctx.lineTo(node.x, node.y);
    ctx.stroke();

    ctx.fillStyle = rgba(p.rgb, 1);
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.fill();

    if (p.flight.audio_confirmed) {
      ctx.strokeStyle = rgba(theme.leaf, 0.85);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r * 2.1, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // The hand, at the current playback time.
  if (s.progress >= 0) {
    const a = toAngle(s.progress);
    ctx.strokeStyle = rgba(theme.leaf, 0.9);
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a) * maxR, cy + Math.sin(a) * maxR);
    ctx.stroke();
  }

  ctx.fillStyle = rgba(theme.muted, 1);
  ctx.beginPath();
  ctx.arc(cx, cy, U * 0.007, 0, Math.PI * 2);
  ctx.fill();
}

// ── Mode 3: Topography ──────────────────────────────────────────────────

/**
 * Accumulate every perch into a coarse density field, plus a colour field
 * so the sand can take on the plumage of whatever lands there.
 *
 * This is the map the mode is actually about: not where birds flew, but
 * where they chose to stop. A Gaussian splat per perch turns scattered
 * points into a continuous surface that contours can be cut from.
 */
function buildField(prepared: Prepared[]): Field {
  const density = new Float32Array(FIELD_COLS * FIELD_ROWS);
  const tint = new Float32Array(FIELD_COLS * FIELD_ROWS * 3);
  const radius = 5;

  for (const p of prepared) {
    const weight = massScale(p.flight.style.mass_g);
    const c = p.perch.x * (FIELD_COLS - 1);
    const r = p.perch.y * (FIELD_ROWS - 1);
    const c0 = Math.max(0, Math.floor(c - radius));
    const c1 = Math.min(FIELD_COLS - 1, Math.ceil(c + radius));
    const r0 = Math.max(0, Math.floor(r - radius));
    const r1 = Math.min(FIELD_ROWS - 1, Math.ceil(r + radius));

    for (let rr = r0; rr <= r1; rr++) {
      for (let cc = c0; cc <= c1; cc++) {
        const d2 = (cc - c) ** 2 + (rr - r) ** 2;
        const g = Math.exp(-d2 / (2 * (radius / 2) ** 2)) * weight;
        if (g < 1e-3) continue;
        const i = rr * FIELD_COLS + cc;
        density[i] += g;
        tint[i * 3] += p.rgb[0] * g;
        tint[i * 3 + 1] += p.rgb[1] * g;
        tint[i * 3 + 2] += p.rgb[2] * g;
      }
    }
  }

  let peak = 0;
  for (let i = 0; i < density.length; i++) if (density[i] > peak) peak = density[i];
  if (peak > 0) {
    for (let i = 0; i < density.length; i++) {
      if (density[i] > 0) {
        tint[i * 3] /= density[i];
        tint[i * 3 + 1] /= density[i];
        tint[i * 3 + 2] /= density[i];
      }
      density[i] /= peak;
    }
  }
  const field: Field = {
    cols: FIELD_COLS,
    rows: FIELD_ROWS,
    density,
    tint,
    wash: null,
    contours: [],
  };
  field.wash = paintWash(field);
  // Cutting contours means walking five thousand cells per level. The field
  // only changes when the day does, so it happens here rather than thirty
  // thousand cell tests deep inside every animation frame.
  field.contours = CONTOUR_LEVELS.map((level) => marchingSquares(field, level));
  return field;
}

/**
 * Paint the density field once, at field resolution, as an image to be
 * scaled up under the contours.
 *
 * This is what gives the map its elevation: contour lines alone are just
 * rings on blank paper. Drawing it as one small bitmap and letting the
 * browser's own smoothing do the interpolation costs a single drawImage
 * per frame, where filling between contour bands would mean stitching
 * marching-squares segments into closed polygons for no better result.
 */
function paintWash(f: Field): HTMLCanvasElement | null {
  if (typeof document === "undefined") return null;
  const c = document.createElement("canvas");
  c.width = f.cols;
  c.height = f.rows;
  const ctx = c.getContext("2d");
  if (!ctx) return null;

  const img = ctx.createImageData(f.cols, f.rows);
  for (let i = 0; i < f.density.length; i++) {
    const d = f.density[i];
    img.data[i * 4] = f.tint[i * 3];
    img.data[i * 4 + 1] = f.tint[i * 3 + 1];
    img.data[i * 4 + 2] = f.tint[i * 3 + 2];
    // Square-rooted so the shallow outskirts stay visible next to the peak.
    img.data[i * 4 + 3] = Math.round(Math.sqrt(d) * 235);
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/**
 * Re-aim the sand at the new field.
 *
 * Grains keep their current position and get a new target, so a change of
 * day makes the sand flow to its new resting place instead of snapping.
 * Targets are rejection-sampled against density, which piles grains up
 * over the busy perches and leaves the dead corners bare.
 */
function scatterGrains(field: Field, existing: Grain[]): Grain[] {
  const out: Grain[] = [];
  const sample = () => {
    // The small floor keeps a thin scatter of sand over the whole tray;
    // without it the bare ground reads as a hole rather than as flat sand.
    for (let tries = 0; tries < 140; tries++) {
      const x = Math.random();
      const y = Math.random();
      const d = fieldAt(field, x, y);
      if (Math.random() < d * 0.97 + 0.006) return { x, y, d };
    }
    return { x: Math.random(), y: Math.random(), d: 0 };
  };

  for (let i = 0; i < GRAIN_COUNT; i++) {
    const t = sample();
    const prev = existing[i];
    const idx = cellIndex(field, t.x, t.y);
    out.push({
      x: prev ? prev.x : t.x,
      y: prev ? prev.y : t.y,
      tx: t.x,
      ty: t.y,
      size: 0.7 + Math.random() * 0.9,
      tone: 0.35 + Math.random() * 0.65,
      rgb: [field.tint[idx * 3], field.tint[idx * 3 + 1], field.tint[idx * 3 + 2]],
    });
  }
  return out;
}

function cellIndex(f: Field, x: number, y: number): number {
  const c = Math.min(f.cols - 1, Math.max(0, Math.round(x * (f.cols - 1))));
  const r = Math.min(f.rows - 1, Math.max(0, Math.round(y * (f.rows - 1))));
  return r * f.cols + c;
}

function fieldAt(f: Field, x: number, y: number): number {
  return f.density[cellIndex(f, x, y)];
}

function drawTopography(ctx: CanvasRenderingContext2D, w: number, h: number, s: Scene) {
  const { theme, field } = s;
  if (!field) return;
  const U = Math.min(w, h);

  // Elevation wash, tinted by whatever plumage settles at each spot.
  if (field.wash) {
    ctx.save();
    ctx.globalAlpha = theme.isDark ? 0.62 : 0.46;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(field.wash, 0, 0, w, h);
    ctx.restore();
  }

  // Contours, each level a shade firmer than the one below it.
  for (let li = 0; li < field.contours.length; li++) {
    const segs = field.contours[li];
    if (!segs.length) continue;
    ctx.strokeStyle = rgba(theme.ink, (theme.isDark ? 0.3 : 0.34) + li * 0.07);
    ctx.lineWidth = li === field.contours.length - 1 ? 1.6 : 0.9;
    ctx.beginPath();
    for (const [x1, y1, x2, y2] of segs) {
      ctx.moveTo(x1 * w, y1 * h);
      ctx.lineTo(x2 * w, y2 * h);
    }
    ctx.stroke();
  }

  // The sand, easing toward its targets. Kinetic while it settles, still
  // once it has, and always still in an export.
  const ease = s.settled ? 1 : 0.06;
  const base: RGB = theme.isDark ? theme.faint : theme.ink;
  for (const g of s.grains) {
    g.x += (g.tx - g.x) * ease;
    g.y += (g.ty - g.y) * ease;
    const local = fieldAt(field, g.x, g.y);
    const hasTint = g.rgb[0] + g.rgb[1] + g.rgb[2] > 0;
    const mixed = mix(base, hasTint ? g.rgb : base, Math.min(1, local * 1.6));
    ctx.fillStyle = rgba(mixed, (theme.isDark ? 0.55 : 0.42) * g.tone * (0.35 + local));
    ctx.fillRect(g.x * w, g.y * h, g.size, g.size);
  }

  // Whichever perch is under the cursor.
  for (const p of s.prepared) {
    if (revealFraction(p.flight, s.progress) <= 0) continue;
    if (p.flight.detection_id !== s.hoveredId && p.flight.detection_id !== s.pinnedId) continue;
    ctx.strokeStyle = rgba(theme.leaf, 0.9);
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(p.perch.x * w, p.perch.y * h, U * 0.024, 0, Math.PI * 2);
    ctx.stroke();
  }
}

/**
 * Contour one level out of the density field.
 *
 * Standard marching squares with linear edge interpolation. Saddle cases
 * (5 and 10) are drawn as the two-segment reading, which for a smooth
 * Gaussian field is the visually correct one often enough not to warrant
 * disambiguating against the cell average.
 */
function marchingSquares(f: Field, level: number): number[][] {
  const segs: number[][] = [];
  const at = (c: number, r: number) => f.density[r * f.cols + c];
  const ix = (c: number) => c / (f.cols - 1);
  const iy = (r: number) => r / (f.rows - 1);

  for (let r = 0; r < f.rows - 1; r++) {
    for (let c = 0; c < f.cols - 1; c++) {
      const tl = at(c, r);
      const tr = at(c + 1, r);
      const br = at(c + 1, r + 1);
      const bl = at(c, r + 1);
      const code =
        (tl > level ? 8 : 0) | (tr > level ? 4 : 0) | (br > level ? 2 : 0) | (bl > level ? 1 : 0);
      if (code === 0 || code === 15) continue;

      const lerp = (a: number, b: number) => (level - a) / (b - a || 1e-6);
      const top = [ix(c) + (ix(c + 1) - ix(c)) * lerp(tl, tr), iy(r)];
      const right = [ix(c + 1), iy(r) + (iy(r + 1) - iy(r)) * lerp(tr, br)];
      const bottom = [ix(c) + (ix(c + 1) - ix(c)) * lerp(bl, br), iy(r + 1)];
      const left = [ix(c), iy(r) + (iy(r + 1) - iy(r)) * lerp(tl, bl)];

      const push = (a: number[], b: number[]) => segs.push([a[0], a[1], b[0], b[1]]);
      switch (code) {
        case 1: case 14: push(left, bottom); break;
        case 2: case 13: push(bottom, right); break;
        case 3: case 12: push(left, right); break;
        case 4: case 11: push(top, right); break;
        case 6: case 9: push(top, bottom); break;
        case 7: case 8: push(left, top); break;
        case 5: push(left, top); push(bottom, right); break;
        case 10: push(top, right); push(left, bottom); break;
      }
    }
  }
  return segs;
}

// ── Preparation and helpers ─────────────────────────────────────────────

function prepare(flight: ArtFlight): Prepared {
  const pts = resample(flight.points, PATH_SAMPLES);

  // Segment length stands in for speed.
  const lens = pts.map((p, i) => {
    const q = pts[Math.min(pts.length - 1, i + 1)];
    return Math.hypot(q.x - p.x, q.y - p.y);
  });
  const maxLen = Math.max(...lens, 1e-6);

  // Cap the stroke against the ground the flight actually covers. A bird
  // that sat on the feeder and shuffled has a real tracked path a few
  // thousandths of a frame wide — narrower than a pigeon's nominal ribbon —
  // and drawing it at full width turns a stationary bird into a scribbled
  // tangle several times its own size. The perch bloom carries that bird;
  // the trace only has to be visible.
  const extent = Math.max(
    Math.max(...pts.map((q) => q.x)) - Math.min(...pts.map((q) => q.x)),
    Math.max(...pts.map((q) => q.y)) - Math.min(...pts.map((q) => q.y))
  );
  const nominal = 0.0062 * massScale(flight.style.mass_g);
  const base = Math.min(nominal, Math.max(extent * 0.3, nominal * 0.12));

  const tracked = flight.path_kind === "tracked";
  const half = pts.map((p, i) => {
    const u = i / (pts.length - 1);
    const endTaper = Math.sin(Math.PI * u) ** 0.4;
    const slow = 1 - 0.5 * (lens[i] / maxLen);
    // A synthesized path has a known dwell in the middle, so the brush
    // loads there. A real track has no such structure — `t` is just how
    // far along the frames you are — so speed alone shapes the stroke.
    const body = tracked
      ? 0.45 + 0.55 * (1 - lens[i] / maxLen)
      : 0.14 + 0.86 * Math.exp(-(((p.t - 0.49) / 0.24) ** 2));
    // Thin the stroke where the path turns hard. A ribbon wider than the
    // corner it is rounding folds through itself and fills as a chevron,
    // and real tracked paths do turn on a dime when a bird lands and
    // pivots. Narrowing there is also what a real brush does.
    return base * endTaper * body * (0.55 + 0.45 * slow) * bendLimit(pts, i);
  });

  // The dwell phase is the real, observed position; use its midpoint as
  // the perch rather than the arbitrary middle of the point list.
  const dwell = flight.points.filter((p) => p.t >= 0.34 && p.t <= 0.64);
  const src = dwell.length ? dwell : flight.points;
  const perch = {
    x: src.reduce((a, p) => a + p.x, 0) / src.length,
    y: src.reduce((a, p) => a + p.y, 0) / src.length,
  };

  return {
    flight,
    pts,
    half,
    perch,
    rgb: hexToRgb(flight.style.primary) ?? [110, 125, 100],
    accent: hexToRgb(flight.style.accent) ?? [60, 70, 55],
    ring: 0,
  };
}

function resample(points: ArtFlight["points"], count: number) {
  const n = points.length;
  if (n === 0) return [{ x: 0.5, y: 0.5, t: 0.5 }];
  if (n < 3) return points.map((p) => ({ x: p.x, y: p.y, t: p.t }));

  const out: { x: number; y: number; t: number }[] = [];
  for (let s = 0; s < count; s++) {
    const scaled = (s / (count - 1)) * (n - 1);
    const i = Math.min(n - 2, Math.floor(scaled));
    const t = scaled - i;
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(n - 1, i + 2)];
    out.push({
      x: catmull(p0.x, p1.x, p2.x, p3.x, t),
      y: catmull(p0.y, p1.y, p2.y, p3.y, t),
      t: p1.t + (p2.t - p1.t) * t,
    });
  }
  return out;
}

function catmull(p0: number, p1: number, p2: number, p3: number, t: number) {
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    0.5 *
    (2 * p1 +
      (-p0 + p2) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
  );
}

/** Body mass on a compressed scale, so an osprey outweighs a chickadee visibly but not absurdly. */
function massScale(grams: number) {
  return Math.min(2.6, Math.max(0.55, Math.pow(Math.max(5, grams) / 40, 0.32)));
}

function hexToRgb(hex: string): RGB | null {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return null;
  let s = m[1];
  if (s.length === 3) s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2];
  const v = parseInt(s, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/**
 * A soft radial dab in one plumage colour, drawn once and reused.
 *
 * Building a fresh CanvasGradient for every perch on every frame is one of
 * the few genuinely expensive things left in the loop; there are at most a
 * few dozen distinct plumage colours on screen, so cache a sprite each and
 * scale it to the bloom's size.
 */
const BLOOM_CACHE = new Map<string, HTMLCanvasElement | null>();
const BLOOM_PX = 96;

function bloomSprite(color: RGB): HTMLCanvasElement | null {
  const key = `${color[0] | 0},${color[1] | 0},${color[2] | 0}`;
  const hit = BLOOM_CACHE.get(key);
  if (hit !== undefined) return hit;

  let made: HTMLCanvasElement | null = null;
  if (typeof document !== "undefined") {
    const c = document.createElement("canvas");
    c.width = c.height = BLOOM_PX;
    const cx = c.getContext("2d");
    if (cx) {
      const half = BLOOM_PX / 2;
      const g = cx.createRadialGradient(half, half, 0, half, half, half);
      g.addColorStop(0, rgba(color, 1));
      g.addColorStop(1, rgba(color, 0));
      cx.fillStyle = g;
      cx.fillRect(0, 0, BLOOM_PX, BLOOM_PX);
      made = c;
    }
  }
  BLOOM_CACHE.set(key, made);
  return made;
}

function mix(a: RGB, b: RGB, k: number): RGB {
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

function rgb(c: RGB) {
  return `rgb(${c[0] | 0}, ${c[1] | 0}, ${c[2] | 0})`;
}

function rgba(c: RGB, a: number) {
  return `rgba(${c[0] | 0}, ${c[1] | 0}, ${c[2] | 0}, ${a})`;
}

const FALLBACK_THEME: Theme = {
  isDark: false,
  paper: [216, 225, 207],
  panel: [230, 235, 218],
  surface: [243, 245, 236],
  ink: [33, 42, 30],
  muted: [94, 106, 85],
  faint: [134, 146, 123],
  line: [210, 220, 196],
  leaf: [53, 101, 68],
};

/**
 * Pull the Field-Guide palette straight out of the stylesheet.
 *
 * index.css keeps bare RGB triplets alongside every hex so Tailwind's
 * alpha modifiers work; those triplets are exactly what a canvas wants,
 * and reading them means the artwork follows Sage and Twilight instead of
 * carrying a second palette that drifts out of step.
 */
function readTheme(): Theme {
  if (typeof document === "undefined") return FALLBACK_THEME;
  const cs = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: RGB): RGB => {
    const parts = cs.getPropertyValue(name).trim().split(/[\s,]+/).map(Number);
    return parts.length === 3 && parts.every((n) => Number.isFinite(n))
      ? (parts as RGB)
      : fallback;
  };
  return {
    isDark: document.documentElement.classList.contains("dark"),
    paper: read("--bg-rgb", FALLBACK_THEME.paper),
    panel: read("--panel-rgb", FALLBACK_THEME.panel),
    surface: read("--card-rgb", FALLBACK_THEME.surface),
    ink: read("--ink-rgb", FALLBACK_THEME.ink),
    muted: read("--muted-rgb", FALLBACK_THEME.muted),
    faint: read("--faint-rgb", FALLBACK_THEME.faint),
    line: read("--line-rgb", FALLBACK_THEME.line),
    leaf: read("--accent-rgb", FALLBACK_THEME.leaf),
  };
}
