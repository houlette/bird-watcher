import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

import type { TavernPatron } from "../lib/api";

/**
 * The common room of The Perch & Flagon, drawn on a 2D canvas.
 *
 * The scene is laid out in normalised coordinates (0-1 on both axes) and
 * scaled to whatever box the page gives it, so the same room works as a
 * full-width panel and as a dock strip at the bottom of a screen.
 *
 * Everything here is decoration over real data. A guest's colour, size and
 * seat all come from the detection that produced them: plumage from the
 * palette the Art page shares, body size from the species' mass, and the
 * seat from the archetype. What is invented is the walking, the furniture
 * and the fire.
 */

export type Phase = "dawn" | "day" | "dusk" | "night";

export interface TavernCanvasHandle {
  /** Play the mishap flourish: the candles lie flat and the room dims. */
  gutter: () => void;
}

export interface TavernCanvasProps {
  /** Who is in the room. Guests who leave this list walk out. */
  patrons: TavernPatron[];
  /** Upgrade ids the house owns, from the tavern state. */
  unlocked: string[];
  phase: Phase;
  /** Bumped by the parent when the theme flips, so the palette is re-read. */
  themeKey: string;
  pickedId: number | null;
  onPick: (patron: TavernPatron | null) => void;
  /** Dock mode: a shorter room with the far corners trimmed. */
  compact?: boolean;
}

type RGB = [number, number, number];

interface Theme {
  isDark: boolean;
  paper: RGB;
  panel: RGB;
  surface: RGB;
  ink: RGB;
  muted: RGB;
  line: RGB;
  leaf: RGB;
}

interface Slot {
  x: number;
  y: number;
  region: string;
  /** Perched guests sit on furniture and hide their legs. */
  perched: boolean;
  taken: number | null;
}

interface Actor {
  patron: TavernPatron;
  slot: Slot;
  state: "entering" | "seated" | "leaving";
  /** Seconds spent in the current state. */
  t: number;
  x: number;
  y: number;
  scale: number;
  facing: 1 | -1;
  seed: number;
  alpha: number;
  /** Countdown to the next hop, in seconds. */
  nextHop: number;
  hop: number;
}

const DOOR = { x: 0.945, y: 0.735 };
/** Room aspect in the page (16:9) and in the dock strip (16:5). */
const ROOM_ASPECT = 16 / 9;
const DOCK_ASPECT = 16 / 5;
/**
 * The band of the room the dock shows. Its height is exactly the ratio of
 * the two aspects, which is what keeps a candle the same size in both, and
 * it starts partway down so the band holds the things worth watching: the
 * fire, the door, the floor and everyone on it.
 */
const DOCK_SPAN = ROOM_ASPECT / DOCK_ASPECT;
const DOCK_TOP = 0.33;
const ENTER_SECONDS = 2.1;
const LEAVE_SECONDS = 1.7;

/**
 * Seat regions, as fractions of the room. The archetype picks the region
 * and the first free slot inside it takes the guest, so two birds never
 * share a stool and the bar fills up left to right the way a bar does.
 */
const SEAT_PLAN: Record<string, { xs: [number, number]; y: number; n: number; perched: boolean }> = {
  hearth: { xs: [0.08, 0.26], y: 0.88, n: 3, perched: false },
  bench: { xs: [0.33, 0.55], y: 0.715, n: 4, perched: true },
  booth: { xs: [0.63, 0.76], y: 0.7, n: 3, perched: true },
  bar: { xs: [0.36, 0.7], y: 0.855, n: 5, perched: true },
  rafters: { xs: [0.28, 0.72], y: 0.235, n: 3, perched: true },
  shadow: { xs: [0.79, 0.88], y: 0.78, n: 3, perched: false },
};

const FALLBACK_THEME: Theme = {
  isDark: false,
  paper: [216, 225, 207],
  panel: [230, 235, 218],
  surface: [243, 245, 236],
  ink: [33, 42, 30],
  muted: [94, 106, 85],
  line: [210, 220, 196],
  leaf: [53, 101, 68],
};

export const TavernCanvas = forwardRef<TavernCanvasHandle, TavernCanvasProps>(
  function TavernCanvas(props, ref) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const boxRef = useRef<HTMLDivElement | null>(null);

    // Everything the loop reads, mirrored so the loop is created once and
    // never torn down by a re-render. Written from an effect, not during
    // render: a render React later discards must not reach the loop.
    const liveRef = useRef(props);
    useEffect(() => {
      liveRef.current = props;
    });

    const themeRef = useRef<Theme>(FALLBACK_THEME);
    useEffect(() => {
      themeRef.current = readTheme();
    }, [props.themeKey]);

    const actorsRef = useRef<Map<number, Actor>>(new Map());
    const slotsRef = useRef<Slot[]>(buildSlots());
    const gutterRef = useRef(0);
    const hitRef = useRef<{ id: number; x: number; y: number; r: number }[]>([]);

    useImperativeHandle(ref, () => ({ gutter: () => (gutterRef.current = 1) }), []);

    // Seat arrivals and send departures to the door. Done in an effect
    // rather than in the loop so the roster is only reconciled when it
    // actually changes.
    useEffect(() => {
      const actors = actorsRef.current;
      const present = new Set(props.patrons.map((p) => p.detection_id));

      for (const patron of props.patrons) {
        if (actors.has(patron.detection_id)) continue;
        const slot = claimSlot(slotsRef.current, patron);
        actors.set(patron.detection_id, {
          patron,
          slot,
          state: "entering",
          t: 0,
          x: DOOR.x,
          y: DOOR.y,
          scale: bodyScale(patron.style.mass_g),
          facing: -1,
          seed: patron.detection_id * 2654435761,
          alpha: 1,
          nextHop: 2 + rand(patron.detection_id, 3) * 6,
          hop: 0,
        });
      }

      for (const [id, actor] of actors) {
        if (present.has(id) || actor.state === "leaving") continue;
        actor.state = "leaving";
        actor.t = 0;
        actor.facing = 1;
        if (actor.slot.taken === id) actor.slot.taken = null;
      }
    }, [props.patrons]);

    // Pointer picking. Registered once; the hit list is rebuilt each frame.
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const pick = (ev: PointerEvent) => {
        const rect = canvas.getBoundingClientRect();
        const px = ev.clientX - rect.left;
        const py = ev.clientY - rect.top;
        let best: { id: number; d: number } | null = null;
        for (const h of hitRef.current) {
          const d = Math.hypot(px - h.x, py - h.y);
          if (d <= h.r * 1.6 && (!best || d < best.d)) best = { id: h.id, d };
        }
        const live = liveRef.current;
        const hit = best ? live.patrons.find((p) => p.detection_id === best!.id) ?? null : null;
        live.onPick(hit);
      };

      canvas.addEventListener("pointerdown", pick);
      return () => canvas.removeEventListener("pointerdown", pick);
    }, []);

    // The one animation loop.
    useEffect(() => {
      const canvas = canvasRef.current;
      const box = boxRef.current;
      if (!canvas || !box) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      let raf = 0;
      let last = performance.now();
      let w = 0;
      let h = 0;

      const resize = () => {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const rect = box.getBoundingClientRect();
        w = Math.max(1, Math.round(rect.width));
        h = Math.max(1, Math.round(rect.height));
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      };

      const observer = new ResizeObserver(resize);
      observer.observe(box);
      resize();

      const frame = (now: number) => {
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        const live = liveRef.current;

        if (gutterRef.current > 0) {
          gutterRef.current = Math.max(0, gutterRef.current - dt / 1.6);
        }

        step(actorsRef.current, dt, now / 1000);
        draw(ctx, w, h, {
          theme: themeRef.current,
          phase: live.phase,
          unlocked: new Set(live.unlocked),
          compact: !!live.compact,
          actors: actorsRef.current,
          pickedId: live.pickedId,
          time: now / 1000,
          gutter: gutterRef.current,
          hits: hitRef.current,
        });

        raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);

      return () => {
        cancelAnimationFrame(raf);
        observer.disconnect();
      };
    }, []);

    const aspect = props.compact ? `${DOCK_ASPECT}` : `${ROOM_ASPECT}`;

    return (
      <div
        ref={boxRef}
        className="relative w-full overflow-hidden rounded-card border border-line"
        style={{ aspectRatio: aspect }}
      >
        <canvas ref={canvasRef} className="block w-full h-full cursor-pointer" />
      </div>
    );
  }
);

// ── Roster ──────────────────────────────────────────────────────────────

function buildSlots(): Slot[] {
  const slots: Slot[] = [];
  for (const [region, plan] of Object.entries(SEAT_PLAN)) {
    for (let i = 0; i < plan.n; i++) {
      const f = plan.n === 1 ? 0.5 : i / (plan.n - 1);
      slots.push({
        x: plan.xs[0] + f * (plan.xs[1] - plan.xs[0]),
        y: plan.y,
        region,
        perched: plan.perched,
        taken: null,
      });
    }
  }
  return slots;
}

/**
 * Give a guest a seat in their own quarter of the room if one is free,
 * and anywhere at all if it is not. A house full of doves should still
 * seat the jay that walks in, even if the booth is spoken for.
 */
function claimSlot(slots: Slot[], patron: TavernPatron): Slot {
  const wanted = patron.archetype.seat;
  const free = slots.filter((s) => s.taken === null);
  const preferred = free.filter((s) => s.region === wanted);
  const pool = preferred.length ? preferred : free.length ? free : slots;
  const pick = pool[Math.floor(rand(patron.detection_id, 11) * pool.length)] ?? pool[0];
  pick.taken = patron.detection_id;
  return pick;
}

function step(actors: Map<number, Actor>, dt: number, time: number) {
  for (const [id, a] of actors) {
    a.t += dt;

    if (a.state === "entering") {
      const k = Math.min(1, a.t / ENTER_SECONDS);
      const e = easeInOut(k);
      const path = walkPath(DOOR, a.slot, e);
      a.x = path.x;
      a.y = path.y;
      a.facing = a.slot.x < DOOR.x ? -1 : 1;
      if (k >= 1) {
        a.state = "seated";
        a.t = 0;
        a.x = a.slot.x;
        a.y = a.slot.y;
      }
    } else if (a.state === "seated") {
      a.nextHop -= dt;
      if (a.nextHop <= 0) {
        a.hop = 1;
        a.nextHop = 3 + rand(id, Math.floor(time)) * 8;
      }
      a.hop = Math.max(0, a.hop - dt / 0.45);
      // A settled bird breathes and shifts; the hop is the bigger move.
      a.x = a.slot.x + Math.sin(time * 0.7 + a.seed % 7) * 0.0016;
      a.y = a.slot.y - Math.sin(a.hop * Math.PI) * 0.035;
    } else {
      const k = Math.min(1, a.t / LEAVE_SECONDS);
      const e = easeInOut(k);
      const path = walkPath(DOOR, a.slot, 1 - e);
      a.x = path.x;
      a.y = path.y;
      a.facing = 1;
      a.alpha = 1 - k * k;
      if (k >= 1) actors.delete(id);
    }
  }
}

/**
 * Door to seat. Guests bound for the rafters fly, in an arc that clears
 * the room; everyone else walks the floor and turns up at their seat.
 */
function walkPath(door: { x: number; y: number }, slot: Slot, k: number) {
  const x = door.x + (slot.x - door.x) * k;
  if (slot.region === "rafters") {
    const lift = Math.sin(k * Math.PI) * 0.12;
    return { x, y: door.y + (slot.y - door.y) * k - lift };
  }
  // Walk along the floor first, then step up to the seat in the last third.
  const floorY = Math.max(door.y, slot.y);
  const rise = k < 0.66 ? 0 : (k - 0.66) / 0.34;
  return { x, y: floorY + (slot.y - floorY) * rise };
}

/**
 * Body size from mass, on a log scale. A Herring Gull is a hundred times
 * a chickadee by weight and cannot be a hundred times the size on screen,
 * but the order has to survive: the gull is unmistakably the big one.
 */
function bodyScale(mass_g: number): number {
  const m = Math.max(6, Math.min(2000, mass_g || 30));
  const k = (Math.log(m) - Math.log(6)) / (Math.log(2000) - Math.log(6));
  return 0.62 + k * 0.95;
}

function easeInOut(k: number) {
  return k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
}

/** Deterministic 0-1 from two integers, so a guest's quirks never change. */
function rand(a: number, b: number): number {
  let h = (a * 374761393 + b * 668265263) >>> 0;
  h = (h ^ (h >>> 13)) * 1274126177;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

// ── Palette ─────────────────────────────────────────────────────────────

function readTheme(): Theme {
  if (typeof document === "undefined") return FALLBACK_THEME;
  const css = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: RGB): RGB => {
    const raw = css.getPropertyValue(name).trim();
    const parts = raw.split(/\s+/).map(Number);
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
    line: read("--line-rgb", FALLBACK_THEME.line),
    leaf: read("--accent-rgb", FALLBACK_THEME.leaf),
  };
}

function rgba(c: RGB, a = 1) {
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
}

function mix(a: RGB, b: RGB, k: number): RGB {
  return [
    Math.round(a[0] + (b[0] - a[0]) * k),
    Math.round(a[1] + (b[1] - a[1]) * k),
    Math.round(a[2] + (b[2] - a[2]) * k),
  ];
}

function hexToRgb(hex: string): RGB {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** The sky in the window, which is the room's only honest clock. */
const SKY: Record<Phase, [RGB, RGB]> = {
  dawn: [[240, 186, 140], [176, 190, 208]],
  day: [[186, 212, 226], [222, 232, 226]],
  dusk: [[214, 138, 92], [92, 96, 128]],
  night: [[26, 32, 54], [12, 16, 30]],
};

// ── Drawing ─────────────────────────────────────────────────────────────

interface Scene {
  theme: Theme;
  phase: Phase;
  unlocked: Set<string>;
  compact: boolean;
  actors: Map<number, Actor>;
  pickedId: number | null;
  time: number;
  gutter: number;
  hits: { id: number; x: number; y: number; r: number }[];
}

function draw(ctx: CanvasRenderingContext2D, w: number, h: number, s: Scene) {
  const { theme } = s;
  // Warmth is the sum of the light the house has bought. A bare room at
  // night is genuinely dim, which is the point of buying candles.
  const lit =
    (s.unlocked.has("tallow_candles") ? 0.14 : 0) +
    (s.unlocked.has("brass_lanterns") ? 0.2 : 0) +
    (s.unlocked.has("suet_hearth") ? 0.1 : 0);
  const night = s.phase === "night" ? 1 : s.phase === "dusk" ? 0.55 : s.phase === "dawn" ? 0.3 : 0;
  const gutter = s.gutter;

  const timber: RGB = mix([92, 66, 44], [58, 42, 30], night * 0.6);
  const wallBase: RGB = theme.isDark ? [70, 56, 45] : [116, 90, 68];
  const wall = mix(wallBase, [30, 24, 20], night * 0.5 - lit * 0.7);
  const floor = mix(theme.isDark ? [78, 56, 38] : [134, 98, 64], [26, 19, 14], night * 0.45 - lit * 0.5);

  ctx.clearRect(0, 0, w, h);
  // Dock mode crops the top of the room rather than squashing it. Scaling
  // the whole scene into a box a third the height turned the fireplace
  // into a letterbox, so the dock cuts the beams off instead and keeps the
  // floor, the fire and the guests at the proportions they have upstairs.
  // The fraction is exactly the difference between the two aspect ratios,
  // which is what makes a shape the same size in both.
  const top = s.compact ? DOCK_TOP : 0;
  const span = s.compact ? DOCK_SPAN : 1;
  const X = (u: number) => u * w;
  const Y = (v: number) => ((v - top) / span) * h;
  const S = Math.min(w, h / span);

  // ── Wall and floor ────────────────────────────────────────────────────
  const wallGrad = ctx.createLinearGradient(0, 0, 0, Y(0.66));
  wallGrad.addColorStop(0, rgba(mix(wall, [0, 0, 0], 0.25)));
  wallGrad.addColorStop(1, rgba(wall));
  ctx.fillStyle = wallGrad;
  ctx.fillRect(0, 0, w, Y(0.66));

  ctx.fillStyle = rgba(floor);
  ctx.fillRect(0, Y(0.66), w, h - Y(0.66));

  // Floorboards, converging slightly so the room has a back to it.
  ctx.strokeStyle = rgba(mix(floor, [0, 0, 0], 0.35), 0.55);
  ctx.lineWidth = Math.max(1, S * 0.003);
  for (let i = 0; i <= 9; i++) {
    const u = i / 9;
    ctx.beginPath();
    ctx.moveTo(X(0.5 + (u - 0.5) * 0.55), Y(0.66));
    ctx.lineTo(X(u * 1.24 - 0.12), h);
    ctx.stroke();
  }
  ctx.strokeStyle = rgba(mix(timber, [0, 0, 0], 0.3), 0.8);
  ctx.lineWidth = Math.max(1.5, S * 0.006);
  ctx.beginPath();
  ctx.moveTo(0, Y(0.66));
  ctx.lineTo(w, Y(0.66));
  ctx.stroke();

  drawWindow(ctx, X, Y, S, s, night);
  drawWallFurniture(ctx, X, Y, S, timber);
  drawRafters(ctx, X, Y, S, timber, s);
  drawHearth(ctx, X, Y, S, s, timber);
  drawFurniture(ctx, X, Y, S, s, timber);
  drawBar(ctx, X, Y, S, s, timber, gutter);
  drawDoor(ctx, X, Y, S, s, timber);

  // ── Guests ────────────────────────────────────────────────────────────
  s.hits.length = 0;
  const ordered = [...s.actors.values()].sort((a, b) => a.y - b.y);
  for (const actor of ordered) {
    const px = X(actor.x);
    const py = Y(actor.y);
    const r = S * 0.042 * actor.scale;
    drawBird(ctx, actor, px, py, r, s);
    s.hits.push({ id: actor.patron.detection_id, x: px, y: py - r * 0.6, r });
  }

  // ── Light over everything ─────────────────────────────────────────────
  if (lit > 0) {
    const glow = ctx.createRadialGradient(X(0.5), Y(0.55), 0, X(0.5), Y(0.55), S * 1.1);
    glow.addColorStop(0, `rgba(255, 208, 140, ${0.16 * (lit / 0.44) * (1 - gutter)})`);
    glow.addColorStop(1, "rgba(255, 208, 140, 0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, w, h);
  }
  if (gutter > 0) {
    ctx.fillStyle = `rgba(8, 10, 14, ${0.42 * gutter})`;
    ctx.fillRect(0, 0, w, h);
  }

  // Dust in the light, which is most of what a still room does.
  ctx.fillStyle = rgba(theme.isDark ? [255, 236, 200] : [255, 250, 232], 0.16);
  for (let i = 0; i < 26; i++) {
    const seed = i * 97;
    const dx = (rand(seed, 1) + s.time * 0.008 * (0.4 + rand(seed, 2))) % 1;
    const dy = (rand(seed, 3) + s.time * 0.004) % 1;
    ctx.fillRect(X(dx), Y(0.1 + dy * 0.75), 1.6, 1.6);
  }

  // Vignette, so the corners of the room fall away.
  const vig = ctx.createRadialGradient(X(0.5), Y(0.5), S * 0.35, X(0.5), Y(0.5), S * 1.05);
  vig.addColorStop(0, "rgba(0,0,0,0)");
  vig.addColorStop(1, `rgba(0,0,0,${0.26 + night * 0.2})`);
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, w, h);
}

type Coord = (u: number) => number;

/**
 * Vertical lengths. Y() maps a position in the room, which stops being the
 * same thing as a height the moment the dock crops the top off: Y(0.4) is
 * where four tenths down lands, not how tall four tenths is.
 */
function lengths(Y: Coord): Coord {
  const origin = Y(0);
  return (dv: number) => Y(dv) - origin;
}

function drawRafters(
  ctx: CanvasRenderingContext2D,
  X: Coord,
  Y: Coord,
  S: number,
  timber: RGB,
  s: Scene
) {
  ctx.fillStyle = rgba(mix(timber, [0, 0, 0], 0.2));
  ctx.fillRect(0, Y(0.2), X(1), S * 0.028);
  ctx.fillStyle = rgba(mix(timber, [0, 0, 0], 0.4), 0.9);
  for (const u of [0.18, 0.42, 0.66, 0.88]) {
    ctx.fillRect(X(u), Y(0), S * 0.02, lengths(Y)(0.2));
  }

  if (s.unlocked.has("rafter_roost")) {
    ctx.fillStyle = "rgba(198, 164, 96, 0.85)";
    for (const u of [0.3, 0.55, 0.72]) {
      ctx.beginPath();
      ctx.ellipse(X(u), Y(0.212), S * 0.03, S * 0.011, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (s.unlocked.has("brass_lanterns")) {
    for (const u of [0.26, 0.5, 0.74]) {
      const sway = Math.sin(s.time * 0.8 + u * 9) * S * 0.004;
      const cx = X(u) + sway;
      const cy = Y(0.3);
      ctx.strokeStyle = rgba(mix(timber, [220, 180, 90], 0.6), 0.9);
      ctx.lineWidth = Math.max(1, S * 0.003);
      ctx.beginPath();
      ctx.moveTo(X(u), Y(0.228));
      ctx.lineTo(cx, cy - S * 0.02);
      ctx.stroke();

      const flick = 0.8 + Math.sin(s.time * 7 + u * 20) * 0.12;
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, S * 0.16);
      glow.addColorStop(0, `rgba(255, 196, 110, ${0.34 * flick * (1 - s.gutter)})`);
      glow.addColorStop(1, "rgba(255, 196, 110, 0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, S * 0.16, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = `rgba(226, 178, 84, ${0.95})`;
      ctx.fillRect(cx - S * 0.014, cy - S * 0.02, S * 0.028, S * 0.04);
      ctx.fillStyle = `rgba(255, 234, 176, ${flick * (1 - s.gutter)})`;
      ctx.fillRect(cx - S * 0.009, cy - S * 0.014, S * 0.018, S * 0.028);
    }
  }
}

/**
 * The two things on the wall that are not for sale: a small framed print
 * and a bundle of herbs hung to dry. Without them the middle of the back
 * wall is a large empty rectangle, and the room reads as a barn.
 */
function drawWallFurniture(
  ctx: CanvasRenderingContext2D,
  X: Coord,
  Y: Coord,
  S: number,
  timber: RGB
) {
  const H = lengths(Y);

  const fx = X(0.35);
  const fy = Y(0.36);
  ctx.fillStyle = rgba(mix(timber, [0, 0, 0], 0.35));
  ctx.fillRect(fx, fy, X(0.07), H(0.12));
  ctx.fillStyle = "rgba(196, 178, 140, 0.35)";
  ctx.fillRect(fx + S * 0.008, fy + S * 0.008, X(0.07) - S * 0.016, H(0.12) - S * 0.016);
  ctx.strokeStyle = "rgba(120, 104, 78, 0.55)";
  ctx.lineWidth = Math.max(1, S * 0.003);
  ctx.beginPath();
  ctx.moveTo(fx + S * 0.012, fy + H(0.09));
  ctx.quadraticCurveTo(fx + X(0.035), fy + H(0.03), fx + X(0.07) - S * 0.012, fy + H(0.08));
  ctx.stroke();

  const hx = X(0.47);
  const hy = Y(0.3);
  ctx.strokeStyle = "rgba(126, 106, 74, 0.9)";
  ctx.lineWidth = Math.max(1.4, S * 0.004);
  for (let i = 0; i < 5; i++) {
    const spread = (i - 2) * S * 0.008;
    ctx.beginPath();
    ctx.moveTo(hx, hy);
    ctx.quadraticCurveTo(hx + spread * 1.6, hy + H(0.05), hx + spread, hy + H(0.1));
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(150, 132, 92, 0.9)";
  ctx.beginPath();
  ctx.arc(hx, hy, S * 0.008, 0, Math.PI * 2);
  ctx.fill();
}

function drawWindow(
  ctx: CanvasRenderingContext2D,
  X: Coord,
  Y: Coord,
  S: number,
  s: Scene,
  night: number
) {
  const x = X(0.56);
  const y = Y(0.27);
  const wdt = X(0.22);
  const hgt = lengths(Y)(0.3);
  const [top, bottom] = SKY[s.phase];

  const sky = ctx.createLinearGradient(0, y, 0, y + hgt);
  sky.addColorStop(0, rgba(top));
  sky.addColorStop(1, rgba(bottom));
  ctx.fillStyle = sky;
  ctx.fillRect(x, y, wdt, hgt);

  if (s.unlocked.has("stained_glass")) {
    // Coloured panes, and the light they throw on the floorboards. The
    // patch of colour tracks the phase, so it moves as the real sun does.
    const panes = ["#8c3b3b", "#3b6a8c", "#8c7a3b", "#3b8c5f", "#6b3b8c", "#8c5a3b"];
    for (let i = 0; i < 6; i++) {
      const px = x + (i % 3) * (wdt / 3);
      const py = y + Math.floor(i / 3) * (hgt / 2);
      ctx.fillStyle = `${panes[i]}bb`;
      ctx.fillRect(px, py, wdt / 3, hgt / 2);
    }
    const drift = s.phase === "dawn" ? -0.08 : s.phase === "dusk" ? 0.08 : 0;
    const patch = ctx.createRadialGradient(
      X(0.52 + drift),
      Y(0.82),
      0,
      X(0.52 + drift),
      Y(0.82),
      S * 0.22
    );
    patch.addColorStop(0, `rgba(190, 120, 190, ${0.2 * (1 - night)})`);
    patch.addColorStop(1, "rgba(190, 120, 190, 0)");
    ctx.fillStyle = patch;
    ctx.fillRect(X(0.28), Y(0.66), X(0.5), lengths(Y)(0.34));
  }

  ctx.strokeStyle = "rgba(48, 34, 24, 0.9)";
  ctx.lineWidth = Math.max(2, S * 0.008);
  ctx.strokeRect(x, y, wdt, hgt);
  ctx.lineWidth = Math.max(1, S * 0.004);
  ctx.beginPath();
  ctx.moveTo(x + wdt / 2, y);
  ctx.lineTo(x + wdt / 2, y + hgt);
  ctx.moveTo(x, y + hgt / 2);
  ctx.lineTo(x + wdt, y + hgt / 2);
  ctx.stroke();
}

function drawFire(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  size: number,
  time: number,
  gutter: number
) {
  const flick = 1 - gutter * 0.8;
  const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, size * 5);
  glow.addColorStop(0, `rgba(255, 168, 66, ${0.4 * flick})`);
  glow.addColorStop(1, "rgba(255, 140, 40, 0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(cx, cy, size * 5, 0, Math.PI * 2);
  ctx.fill();

  for (let i = 0; i < 3; i++) {
    const wob = Math.sin(time * (5 + i * 1.7) + i) * size * 0.22;
    const tall = size * (1.5 + Math.sin(time * (6 + i) + i * 2) * 0.3) * flick;
    ctx.fillStyle = [
      `rgba(255, 108, 40, ${0.9 * flick})`,
      `rgba(255, 168, 60, ${0.9 * flick})`,
      `rgba(255, 226, 150, ${0.9 * flick})`,
    ][i];
    ctx.beginPath();
    ctx.moveTo(cx - size * (0.6 - i * 0.16), cy);
    ctx.quadraticCurveTo(cx + wob, cy - tall, cx + size * (0.6 - i * 0.16), cy);
    ctx.closePath();
    ctx.fill();
  }

  // Embers, which do most of the work of making a fire look alive.
  for (let i = 0; i < 5; i++) {
    const t = (time * 0.5 + i * 0.2) % 1;
    ctx.fillStyle = `rgba(255, 190, 110, ${(1 - t) * 0.7 * flick})`;
    ctx.fillRect(
      cx + Math.sin(time * 2 + i * 3) * size * 0.7,
      cy - t * size * 3.2,
      size * 0.12,
      size * 0.12
    );
  }
}

function drawHearth(
  ctx: CanvasRenderingContext2D,
  X: Coord,
  Y: Coord,
  S: number,
  s: Scene,
  timber: RGB
) {
  const x = X(0.06);
  const y = Y(0.42);
  const wdt = X(0.2);
  const hgt = lengths(Y)(0.4);

  ctx.fillStyle = "rgba(74, 68, 62, 0.95)";
  ctx.fillRect(x, y, wdt, hgt);
  ctx.fillStyle = "rgba(28, 24, 22, 0.95)";
  ctx.beginPath();
  ctx.moveTo(x + wdt * 0.14, y + hgt);
  ctx.lineTo(x + wdt * 0.14, y + hgt * 0.42);
  ctx.quadraticCurveTo(x + wdt * 0.5, y + hgt * 0.08, x + wdt * 0.86, y + hgt * 0.42);
  ctx.lineTo(x + wdt * 0.86, y + hgt);
  ctx.closePath();
  ctx.fill();

  // Stonework.
  ctx.strokeStyle = "rgba(120, 112, 104, 0.5)";
  ctx.lineWidth = 1;
  for (let r = 0; r < 5; r++) {
    const ry = y + (r / 5) * hgt;
    ctx.beginPath();
    ctx.moveTo(x, ry);
    ctx.lineTo(x + wdt, ry);
    ctx.stroke();
  }

  ctx.fillStyle = rgba(mix(timber, [0, 0, 0], 0.15));
  ctx.fillRect(x - X(0.012), y - S * 0.018, wdt + X(0.024), S * 0.022);

  drawFire(ctx, x + wdt * 0.5, y + hgt * 0.94, S * 0.045, s.time, s.gutter);

  if (s.unlocked.has("suet_hearth")) {
    drawFire(ctx, X(0.3), Y(0.79), S * 0.015, s.time * 1.3, s.gutter);
    ctx.fillStyle = "rgba(226, 214, 180, 0.9)";
    ctx.fillRect(X(0.29), Y(0.8), S * 0.024, S * 0.02);
  }

  if (s.unlocked.has("acoustic_perches")) {
    ctx.strokeStyle = "rgba(176, 138, 92, 0.95)";
    ctx.lineWidth = Math.max(2, S * 0.006);
    for (const v of [0.36, 0.3]) {
      ctx.beginPath();
      ctx.moveTo(X(0.05), Y(v));
      ctx.lineTo(X(0.3), Y(v - 0.015));
      ctx.stroke();
    }
  }

  if (s.unlocked.has("bard_in_residence")) {
    // The resident player: a hunched shape with a lute, by the fire.
    const bx = X(0.3);
    const by = Y(0.76);
    ctx.fillStyle = "rgba(62, 46, 74, 0.95)";
    ctx.beginPath();
    ctx.ellipse(bx, by, S * 0.022, S * 0.03, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(bx - S * 0.004, by - S * 0.034, S * 0.014, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(198, 156, 96, 0.95)";
    ctx.lineWidth = Math.max(1.4, S * 0.004);
    ctx.beginPath();
    ctx.ellipse(bx + S * 0.018, by + S * 0.004, S * 0.012, S * 0.016, 0.5, 0, Math.PI * 2);
    ctx.moveTo(bx + S * 0.026, by - S * 0.006);
    ctx.lineTo(bx + S * 0.046, by - S * 0.03);
    ctx.stroke();
    // A note or two, on the beat.
    const beat = (s.time * 0.9) % 1;
    ctx.fillStyle = `rgba(255, 230, 180, ${(1 - beat) * 0.8})`;
    ctx.font = `${S * 0.03}px serif`;
    ctx.fillText("♪", bx + S * 0.05, by - S * 0.04 - beat * S * 0.08);
  }
}

function drawFurniture(
  ctx: CanvasRenderingContext2D,
  X: Coord,
  Y: Coord,
  S: number,
  s: Scene,
  timber: RGB
) {
  const oak = s.unlocked.has("oak_tables");
  const top: RGB = oak ? mix(timber, [188, 140, 84], 0.55) : mix(timber, [150, 128, 100], 0.3);

  // The long bench along the back wall, which is where the regulars are.
  ctx.fillStyle = rgba(mix(top, [0, 0, 0], 0.25));
  ctx.fillRect(X(0.3), Y(0.73), X(0.28), S * 0.016);
  ctx.fillStyle = rgba(mix(top, [0, 0, 0], 0.45));
  ctx.fillRect(X(0.31), Y(0.746), S * 0.014, S * 0.05);
  ctx.fillRect(X(0.56), Y(0.746), S * 0.014, S * 0.05);

  // A table in the middle of the floor. Round and oiled once bought.
  const tx = X(0.44);
  const ty = Y(0.8);
  ctx.fillStyle = rgba(top);
  if (oak) {
    ctx.beginPath();
    ctx.ellipse(tx, ty, S * 0.09, S * 0.028, 0, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.fillRect(tx - S * 0.085, ty - S * 0.014, S * 0.17, S * 0.026);
  }
  ctx.fillStyle = rgba(mix(top, [0, 0, 0], 0.45));
  ctx.fillRect(tx - S * 0.012, ty, S * 0.024, S * 0.07);

  if (s.unlocked.has("tallow_candles")) {
    for (let i = 0; i < 3; i++) {
      const cx = tx - S * 0.05 + i * S * 0.05;
      const cy = ty - S * 0.016;
      ctx.fillStyle = "rgba(244, 236, 214, 0.95)";
      ctx.fillRect(cx - S * 0.004, cy - S * 0.03, S * 0.008, S * 0.03);
      // The gutter effect lays the flames flat rather than snuffing them.
      const lean = s.gutter * S * 0.02;
      const flick = 1 - s.gutter * 0.7;
      ctx.fillStyle = `rgba(255, 214, 140, ${flick})`;
      ctx.beginPath();
      ctx.ellipse(
        cx + lean,
        cy - S * 0.036,
        S * 0.005 * flick,
        S * 0.011 * flick,
        s.gutter * 1.2,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
  }

  if (s.unlocked.has("corner_booth")) {
    const H = lengths(Y);
    const bx = X(0.61);
    const by = Y(0.56);
    ctx.fillStyle = rgba(mix(timber, [0, 0, 0], 0.35));
    ctx.fillRect(bx, by, X(0.19), H(0.2));
    ctx.fillStyle = "rgba(112, 46, 52, 0.9)";
    ctx.fillRect(bx + S * 0.01, by + S * 0.01, X(0.17), H(0.14));
    // A drawn-back curtain, because the booth is always spoken for.
    ctx.fillStyle = "rgba(78, 32, 38, 0.95)";
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.quadraticCurveTo(bx + S * 0.03, by + H(0.12), bx + S * 0.006, by + H(0.2));
    ctx.lineTo(bx, by + H(0.2));
    ctx.closePath();
    ctx.fill();
  }
}

function drawBar(
  ctx: CanvasRenderingContext2D,
  X: Coord,
  Y: Coord,
  S: number,
  s: Scene,
  timber: RGB,
  gutter: number
) {
  const x = X(0.32);
  const y = Y(0.88);
  const wdt = X(0.46);

  if (s.unlocked.has("thistle_pantry")) {
    // Shelves of jars behind the bar.
    ctx.fillStyle = rgba(mix(timber, [0, 0, 0], 0.3));
    ctx.fillRect(X(0.8), Y(0.36), X(0.16), S * 0.01);
    ctx.fillRect(X(0.8), Y(0.46), X(0.16), S * 0.01);
    const jars = ["#c8b58a", "#8a9c6a", "#b98a5c", "#7d8fa0"];
    for (let i = 0; i < 8; i++) {
      const jx = X(0.81) + (i % 4) * X(0.037);
      const jy = i < 4 ? Y(0.36) : Y(0.46);
      ctx.fillStyle = jars[i % 4];
      ctx.fillRect(jx, jy - S * 0.03, S * 0.02, S * 0.03);
    }
  }

  if (s.unlocked.has("sunflower_cask")) {
    const cx = X(0.86);
    const cy = Y(0.78);
    ctx.fillStyle = rgba(mix(timber, [140, 96, 56], 0.5));
    ctx.beginPath();
    ctx.ellipse(cx, cy, S * 0.038, S * 0.052, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(60, 52, 44, 0.9)";
    ctx.lineWidth = Math.max(1.5, S * 0.005);
    for (const dy of [-0.02, 0.02]) {
      ctx.beginPath();
      ctx.ellipse(cx, cy + S * dy, S * 0.036, S * 0.012, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  ctx.fillStyle = rgba(mix(timber, [0, 0, 0], 0.42));
  ctx.fillRect(x, y, wdt, Y(1) - y);
  ctx.fillStyle = rgba(mix(timber, [186, 138, 82], 0.5));
  ctx.fillRect(x - S * 0.008, y - S * 0.012, wdt + S * 0.016, S * 0.016);

  if (s.unlocked.has("tallow_candles")) {
    for (let i = 0; i < 4; i++) {
      const cx = x + wdt * (0.16 + i * 0.23);
      const flick = (1 - gutter * 0.75) * (0.85 + Math.sin(s.time * 9 + i * 3) * 0.15);
      ctx.fillStyle = "rgba(240, 232, 208, 0.95)";
      ctx.fillRect(cx, y - S * 0.05, S * 0.007, S * 0.038);
      ctx.fillStyle = `rgba(255, 208, 128, ${flick})`;
      ctx.beginPath();
      ctx.ellipse(
        cx + S * 0.0035 + gutter * S * 0.02,
        y - S * 0.056,
        S * 0.005,
        S * 0.011 * flick,
        gutter * 1.3,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
  }
}

function drawDoor(
  ctx: CanvasRenderingContext2D,
  X: Coord,
  Y: Coord,
  S: number,
  s: Scene,
  timber: RGB
) {
  const x = X(0.9);
  const y = Y(0.4);
  const wdt = X(0.09);
  const hgt = lengths(Y)(0.36);

  ctx.fillStyle = rgba(mix(timber, [0, 0, 0], 0.5));
  ctx.fillRect(x, y, wdt, hgt);
  ctx.strokeStyle = "rgba(196, 158, 96, 0.8)";
  ctx.lineWidth = Math.max(1.5, S * 0.005);
  ctx.strokeRect(x + S * 0.008, y + S * 0.008, wdt - S * 0.016, hgt - S * 0.016);
  ctx.fillStyle = "rgba(214, 178, 110, 0.9)";
  ctx.beginPath();
  ctx.arc(x + S * 0.018, y + hgt * 0.55, S * 0.008, 0, Math.PI * 2);
  ctx.fill();

  if (s.unlocked.has("guest_ledger")) {
    // The book on its stand, just inside the door.
    ctx.fillStyle = rgba(mix(timber, [0, 0, 0], 0.35));
    ctx.fillRect(X(0.855), Y(0.78), S * 0.008, S * 0.06);
    ctx.fillStyle = "rgba(126, 62, 54, 0.95)";
    ctx.fillRect(X(0.84), Y(0.76), S * 0.04, S * 0.022);
    ctx.fillStyle = "rgba(232, 226, 208, 0.95)";
    ctx.fillRect(X(0.842), Y(0.762), S * 0.036, S * 0.014);
  }

  if (s.unlocked.has("painted_sign")) {
    const sway = Math.sin(s.time * 0.6) * 0.03;
    ctx.save();
    ctx.translate(X(0.945), Y(0.3));
    ctx.rotate(sway);
    ctx.fillStyle = "rgba(38, 30, 24, 0.95)";
    ctx.fillRect(-S * 0.07, 0, S * 0.14, S * 0.06);
    ctx.strokeStyle = "rgba(214, 178, 96, 0.95)";
    ctx.lineWidth = Math.max(1.2, S * 0.0035);
    ctx.strokeRect(-S * 0.066, S * 0.004, S * 0.132, S * 0.052);
    ctx.fillStyle = "rgba(226, 192, 112, 0.95)";
    ctx.font = `600 ${S * 0.022}px Newsreader, serif`;
    ctx.textAlign = "center";
    ctx.fillText("Perch", 0, S * 0.028);
    ctx.fillText("& Flagon", 0, S * 0.05);
    ctx.textAlign = "start";
    ctx.restore();
  }
}

/**
 * One guest. Body, head, beak, tail and a wing, all from two plumage
 * colours: the species' own primary and its accent note.
 */
function drawBird(
  ctx: CanvasRenderingContext2D,
  a: Actor,
  px: number,
  py: number,
  r: number,
  s: Scene
) {
  const primary = hexToRgb(a.patron.style.primary || "#7d8d99");
  const accent = hexToRgb(a.patron.style.accent || "#333333");
  const picked = s.pickedId === a.patron.detection_id;
  const stranger = a.patron.species === null;

  ctx.save();
  ctx.globalAlpha = a.alpha * (a.patron.stale ? 0.55 : 1);
  ctx.translate(px, py);
  ctx.scale(a.facing, 1);

  // Shadow, which is what keeps the bird on the floor rather than over it,
  // and fades out as they climb so nothing casts one onto the back wall.
  const grounded = Math.max(0, Math.min(1, (a.y - 0.55) / 0.12));
  if (grounded > 0) {
    ctx.fillStyle = `rgba(0, 0, 0, ${0.28 * grounded})`;
    ctx.beginPath();
    ctx.ellipse(0, 0, r * 0.8, r * 0.22, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  const lift = a.slot.perched && a.state === "seated" ? r * 0.1 : 0;
  ctx.translate(0, -lift);

  if (picked) {
    ctx.strokeStyle = rgba(s.theme.leaf, 0.9);
    ctx.lineWidth = Math.max(1.5, r * 0.1);
    ctx.beginPath();
    ctx.ellipse(0, -r * 0.75, r * 1.15, r * 1.35, 0, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Legs, unless perched on furniture where they would poke through it.
  if (!a.slot.perched || a.state !== "seated") {
    ctx.strokeStyle = rgba(mix(accent, [30, 24, 20], 0.4));
    ctx.lineWidth = Math.max(1, r * 0.09);
    const stride = a.state === "entering" ? Math.sin(a.t * 14) * r * 0.18 : 0;
    ctx.beginPath();
    ctx.moveTo(-r * 0.12 + stride, -r * 0.05);
    ctx.lineTo(-r * 0.12 + stride, 0);
    ctx.moveTo(r * 0.12 - stride, -r * 0.05);
    ctx.lineTo(r * 0.12 - stride, 0);
    ctx.stroke();
  }

  // Tail.
  ctx.fillStyle = rgba(mix(primary, [0, 0, 0], 0.25));
  ctx.beginPath();
  ctx.moveTo(-r * 0.5, -r * 0.7);
  ctx.lineTo(-r * 1.5, -r * 0.5 - Math.sin(s.time * 2 + a.seed % 5) * r * 0.06);
  ctx.lineTo(-r * 0.5, -r * 0.35);
  ctx.closePath();
  ctx.fill();

  // Body.
  ctx.fillStyle = rgba(primary);
  ctx.beginPath();
  ctx.ellipse(0, -r * 0.62, r * 0.72, r * 0.6, -0.12, 0, Math.PI * 2);
  ctx.fill();

  // Wing, which flicks when the bird hops.
  ctx.fillStyle = rgba(mix(primary, accent, 0.45));
  ctx.save();
  ctx.rotate(-a.hop * 0.5);
  ctx.beginPath();
  ctx.ellipse(-r * 0.1, -r * 0.6, r * 0.44, r * 0.26, 0.25, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Head and beak.
  const headX = r * 0.52;
  const headY = -r * 1.12;
  ctx.fillStyle = rgba(primary);
  ctx.beginPath();
  ctx.arc(headX, headY, r * 0.38, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = rgba(accent);
  ctx.beginPath();
  ctx.arc(headX - r * 0.1, headY - r * 0.06, r * 0.2, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "rgba(226, 184, 92, 0.95)";
  ctx.beginPath();
  ctx.moveTo(headX + r * 0.3, headY);
  ctx.lineTo(headX + r * 0.72, headY + r * 0.08);
  ctx.lineTo(headX + r * 0.3, headY + r * 0.16);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "rgba(20, 18, 16, 0.95)";
  ctx.beginPath();
  ctx.arc(headX + r * 0.1, headY - r * 0.06, r * 0.075, 0, Math.PI * 2);
  ctx.fill();

  // A stranger keeps their hood up: the classifier would not name them,
  // and the room should not pretend to know more than the classifier did.
  if (stranger) {
    ctx.fillStyle = "rgba(28, 26, 34, 0.72)";
    ctx.beginPath();
    ctx.moveTo(-r * 0.8, -r * 0.2);
    ctx.quadraticCurveTo(-r * 0.2, -r * 1.9, headX + r * 0.34, headY - r * 0.1);
    ctx.quadraticCurveTo(headX + r * 0.1, headY + r * 0.5, r * 0.2, -r * 0.15);
    ctx.closePath();
    ctx.fill();
  }

  // Heard as well as seen: a small ring, the same idea as the feed's badge.
  if (a.patron.audio_confirmed) {
    ctx.strokeStyle = rgba(s.theme.leaf, 0.5 + Math.sin(s.time * 3 + a.seed % 9) * 0.2);
    ctx.lineWidth = Math.max(1, r * 0.06);
    for (const k of [1.25, 1.6]) {
      ctx.beginPath();
      ctx.arc(headX, headY, r * k, -0.9, 0.5);
      ctx.stroke();
    }
  }

  ctx.restore();
}
