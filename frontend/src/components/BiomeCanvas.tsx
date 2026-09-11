import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";

import type { BiomeGarden, BiomePlant, BiomePollinator } from "../lib/api";

export interface BiomeCanvasHandle {
  /** Render the garden at print resolution and hand back a PNG blob. */
  exportPNG: (width?: number) => Promise<Blob | null>;
}

export interface BiomeCanvasProps {
  plants: BiomePlant[];
  pollinators: BiomePollinator[];
  sun: BiomeGarden["sun"];
  /**
   * Playback position as a fraction of the local day, or -1 for "fully
   * grown". A ref rather than a value for the same reason the Art page
   * uses one: playback advances every frame, and a number prop would tear
   * down this component's animation loop sixty times a second.
   */
  progressRef: React.MutableRefObject<number>;
  /** Species name of the plant the page has pinned, or null. */
  pickedSpecies: string | null;
  onHover: (plant: BiomePlant | null) => void;
  onPick: (plant: BiomePlant | null) => void;
  /** Bumped by the parent when the theme flips, so the palette is re-read. */
  themeKey: string;
  /** Pollinators are the loudest thing on screen; the page can stop them. */
  showPollinators: boolean;
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

/** One drawn stem segment, in skeleton space with the base at the origin. */
interface Segment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Relative width, 1 at the trunk. */
  w: number;
  /** Path distance from the root at each end, normalised to [0, 1]. */
  d0: number;
  d1: number;
}

interface Ornament {
  x: number;
  y: number;
  angle: number;
  kind: "leaf" | "bloom";
  d: number;
  /** Stable per-ornament jitter, so no two leaves are identical. */
  wobble: number;
}

interface Skeleton {
  segments: Segment[];
  ornaments: Ornament[];
  /** Horizontal extent either side of the base, in units of plant height. */
  left: number;
  right: number;
}

interface Placed {
  plant: BiomePlant;
  skeleton: Skeleton;
  /** Base position as fractions of the canvas. */
  x: number;
  y: number;
  /** How far back in the bed, 0 at the front. Drives size and draw order. */
  depth: number;
  /** Plant height as a fraction of the canvas height. */
  height: number;
  sprite: HTMLCanvasElement | null;
  spriteGrowth: number;
  spriteKey: string;
}

interface Mote {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
}

/**
 * Growth is redrawn into a per-plant sprite, not stroked live. A busy day
 * is twenty-five plants of up to seven hundred segments each, and stroking
 * every one of those every frame is the difference between a terrarium and
 * a slideshow. The sprite is rebuilt when growth moves by this much, which
 * is about fifty redraws across a full day of playback.
 */
const GROWTH_STEP = 0.02;

/** Widest and narrowest a plant gets, as a fraction of canvas height. */
const HEIGHT_SCALE = 0.5;

/** The band of canvas the bases sit in, front row to back row. */
const BED_TOP = 0.62;
const BED_BOTTOM = 0.93;

/**
 * Minimum gap between two bases, as a fraction of width.
 *
 * Wide enough that a full garden uses the whole bed. Pitch alone crowds
 * everything to the right, because most of what a yard hears sings above
 * 2 kHz and only doves and crows sit at the bottom of the ruler.
 */
const MIN_GAP = 0.055;

const MOTE_COUNT = 44;

/** Frequency ticks along the ground, in hertz. A spectrogram's axis. */
const HZ_TICKS = [500, 1000, 2000, 4000, 8000];
const HZ_FLOOR = 400;
const HZ_CEILING = 8000;

export const BiomeCanvas = forwardRef<BiomeCanvasHandle, BiomeCanvasProps>(function BiomeCanvas(
  props,
  ref
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  // Everything the render loop needs, mirrored so the loop is set up once.
  // Synced from an effect rather than during render: a render React later
  // discards must not have written anything the loop can see.
  const liveRef = useRef(props);
  useEffect(() => {
    liveRef.current = props;
  });

  const themeRef = useRef<Theme>(FALLBACK_THEME);
  const hoverRef = useRef<string | null>(null);
  const motesRef = useRef<Mote[]>([]);

  // Expanding an L-system and walking it with a turtle is the expensive
  // part and depends only on the data, so it happens here rather than per
  // frame. A garden of twenty-five plants takes a few milliseconds.
  const placed = useMemo(() => layout(props.plants), [props.plants]);
  const placedRef = useRef(placed);
  useEffect(() => {
    placedRef.current = placed;
  }, [placed]);

  // A theme flip invalidates every sprite: the stems are drawn against the
  // ground colour and picked out with the ink colour.
  useEffect(() => {
    themeRef.current = readTheme();
    for (const p of placedRef.current) p.sprite = null;
  }, [props.themeKey]);

  useImperativeHandle(
    ref,
    () => ({
      exportPNG: (width = 3840) =>
        new Promise((resolve) => {
          const off = document.createElement("canvas");
          off.width = width;
          off.height = Math.round((width * 9) / 16);
          const ctx = off.getContext("2d");
          if (!ctx) return resolve(null);
          // A fresh copy of the layout, so the export's sprites are built
          // at print resolution and the on-screen ones are left alone.
          const forPrint = layout(liveRef.current.plants);
          drawScene(ctx, off.width, off.height, {
            placed: forPrint,
            theme: themeRef.current,
            sun: liveRef.current.sun,
            pollinators: liveRef.current.showPollinators
              ? liveRef.current.pollinators
              : [],
            progress: liveRef.current.progressRef.current,
            hovered: null,
            picked: liveRef.current.pickedSpecies,
            motes: [],
            time: 0,
            still: true,
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
      // Sprites are rendered at the pixel size they are drawn at, so a
      // resize retires all of them rather than scaling them up blurry.
      for (const p of placedRef.current) p.sprite = null;
      motesRef.current = makeMotes(canvas.width, canvas.height);
    };

    const ro = new ResizeObserver(resize);
    ro.observe(box);
    resize();

    const hitTest = (clientX: number, clientY: number): BiomePlant | null => {
      const rect = canvas.getBoundingClientRect();
      const mx = (clientX - rect.left) / rect.width;
      const my = (clientY - rect.top) / rect.height;

      // Front to back, so a plant standing in front of another wins the
      // click the way it wins the pixels.
      let best: BiomePlant | null = null;
      let bestDepth = Infinity;
      for (const p of placedRef.current) {
        const h = p.height;
        const left = p.x + p.skeleton.left * h * (cssH / cssW);
        const right = p.x + p.skeleton.right * h * (cssH / cssW);
        if (mx < left - 0.01 || mx > right + 0.01) continue;
        if (my > p.y + 0.02 || my < p.y - h - 0.02) continue;
        if (p.depth < bestDepth) {
          bestDepth = p.depth;
          best = p.plant;
        }
      }
      return best;
    };

    const onMove = (e: PointerEvent) => {
      const hit = hitTest(e.clientX, e.clientY);
      const name = hit?.species ?? null;
      if (name !== hoverRef.current) {
        hoverRef.current = name;
        liveRef.current.onHover(hit);
        canvas.style.cursor = hit ? "pointer" : "default";
      }
    };
    const onLeave = () => {
      if (hoverRef.current !== null) {
        hoverRef.current = null;
        liveRef.current.onHover(null);
      }
    };
    const onClick = (e: PointerEvent) => {
      const hit = hitTest(e.clientX, e.clientY);
      liveRef.current.onPick(hit);
    };

    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("click", onClick as EventListener);

    const ctx = canvas.getContext("2d");
    const start = performance.now();

    const frame = () => {
      raf = requestAnimationFrame(frame);
      if (!ctx) return;
      const live = liveRef.current;
      drawScene(ctx, canvas.width, canvas.height, {
        placed: placedRef.current,
        theme: themeRef.current,
        sun: live.sun,
        pollinators: live.showPollinators ? live.pollinators : [],
        progress: live.progressRef.current,
        hovered: hoverRef.current,
        picked: live.pickedSpecies,
        motes: motesRef.current,
        time: (performance.now() - start) / 1000,
        still: false,
      });
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("click", onClick as EventListener);
    };
  }, []);

  return (
    <div ref={boxRef} className="w-full rounded-card overflow-hidden border border-line">
      <canvas ref={canvasRef} className="block w-full" />
    </div>
  );
});

// ── Layout ──────────────────────────────────────────────────────────────

/**
 * Place every plant in the bed.
 *
 * The x axis is pitch, low on the left and high on the right, which is the
 * axis a spectrogram uses and the reason the garden reads as one. Depth
 * within the bed is seeded per species, so a plant keeps its place in the
 * row from one day to the next.
 */
function layout(plants: BiomePlant[]): Placed[] {
  const placed: Placed[] = plants.map((plant) => {
    const depth = hash01(plant.seed);
    return {
      plant,
      skeleton: buildSkeleton(plant),
      x: 0.06 + plant.pitch * 0.88,
      y: BED_TOP + depth * (BED_BOTTOM - BED_TOP),
      depth,
      // Plants further back stand smaller, which is the only perspective
      // cue a flat canvas gets.
      height: HEIGHT_SCALE * plant.reach * (1.05 - 0.3 * depth),
      sprite: null,
      spriteGrowth: -1,
      spriteKey: "",
    };
  });

  spreadOut(placed);
  // Back row first, so the front of the bed overlaps it rather than the
  // other way round.
  placed.sort((a, b) => b.depth - a.depth);
  return placed;
}

/**
 * Push bases apart until none are closer than MIN_GAP.
 *
 * Pitch alone stacks species that call at the same frequency into one
 * spot: this yard hears three different sparrows within a few hundred
 * hertz of each other. A handful of relaxation passes separates them
 * without losing the ordering.
 */
function spreadOut(placed: Placed[]) {
  const order = [...placed].sort((a, b) => a.x - b.x);
  for (let pass = 0; pass < 24; pass++) {
    let moved = false;
    for (let i = 1; i < order.length; i++) {
      const gap = order[i].x - order[i - 1].x;
      if (gap >= MIN_GAP) continue;
      const push = (MIN_GAP - gap) / 2;
      order[i - 1].x -= push;
      order[i].x += push;
      moved = true;
    }
    for (const p of order) p.x = Math.max(0.05, Math.min(0.95, p.x));
    if (!moved) break;
  }
}

// ── The L-system ────────────────────────────────────────────────────────

/**
 * Expand a plant's rules and walk the result with a turtle.
 *
 * The alphabet is the one pipeline/biome.py documents: F draws a segment,
 * X is a bud that draws nothing, + and - turn, [ and ] branch, L places a
 * leaf and O a bloom. The result is normalised so the plant is exactly one
 * unit tall with its base at the origin, which is what lets the caller
 * scale it to any canvas without re-walking it.
 */
function buildSkeleton(plant: BiomePlant): Skeleton {
  let s = plant.axiom;
  for (let i = 0; i < plant.depth; i++) {
    let out = "";
    for (const ch of s) out += plant.rules[ch] ?? ch;
    s = out;
    // The server caps depth against the same budget, so this should never
    // bite. It is here because a rules edit that forgets the cap should
    // cost a stunted plant and not a frozen tab.
    if (s.length > 40000) break;
  }

  const angle = (plant.angle * Math.PI) / 180;
  const segments: Segment[] = [];
  const ornaments: Ornament[] = [];
  const stack: { x: number; y: number; a: number; w: number; len: number; d: number }[] = [];

  let x = 0;
  let y = 0;
  let a = -Math.PI / 2; // straight up
  let w = 1;
  let len = 1;
  let d = 0;
  let maxD = 0;
  let minX = 0;
  let maxX = 0;
  let minY = 0;
  let n = 0;

  for (const ch of s) {
    switch (ch) {
      case "F": {
        // A per-segment waver, seeded by the species and the position in
        // the string, so a fern is a fern and not a diagram of one.
        const bend = (hash01(plant.seed + n * 2654435761) - 0.5) * 0.22;
        a += bend;
        const nx = x + Math.cos(a) * len;
        const ny = y + Math.sin(a) * len;
        segments.push({ x1: x, y1: y, x2: nx, y2: ny, w, d0: d, d1: d + len });
        x = nx;
        y = ny;
        d += len;
        maxD = Math.max(maxD, d);
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        n++;
        break;
      }
      case "+":
        a += angle;
        break;
      case "-":
        a -= angle;
        break;
      case "[":
        stack.push({ x, y, a, w, len, d });
        // Branches are thinner and shorter than what they grew from, which
        // is most of what makes a turtle walk look like a plant.
        w *= 0.7;
        len *= 0.78;
        break;
      case "]": {
        const top = stack.pop();
        if (top) ({ x, y, a, w, len, d } = top);
        break;
      }
      case "L":
      case "O":
        ornaments.push({
          x,
          y,
          angle: a,
          kind: ch === "L" ? "leaf" : "bloom",
          d,
          wobble: hash01(plant.seed + ornaments.length * 40503),
        });
        n++;
        break;
      default:
        break;
    }
  }

  // Normalise to one unit tall with the base at the origin. Height is
  // measured from the base upward: a branch that dips below the base is
  // part of the plant's spread, not part of its height.
  const height = Math.max(1e-6, -minY);
  const scale = 1 / height;
  for (const seg of segments) {
    seg.x1 *= scale;
    seg.y1 *= scale;
    seg.x2 *= scale;
    seg.y2 *= scale;
    seg.d0 /= maxD || 1;
    seg.d1 /= maxD || 1;
  }
  for (const o of ornaments) {
    o.x *= scale;
    o.y *= scale;
    o.d /= maxD || 1;
  }

  return {
    segments,
    ornaments,
    left: minX * scale,
    right: maxX * scale,
  };
}

// ── Drawing ─────────────────────────────────────────────────────────────

interface SceneState {
  placed: Placed[];
  theme: Theme;
  sun: BiomeGarden["sun"];
  pollinators: BiomePollinator[];
  progress: number;
  hovered: string | null;
  picked: string | null;
  motes: Mote[];
  time: number;
  /** Export renders one still frame with no sway and no motes. */
  still: boolean;
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  state: SceneState
) {
  const { theme, sun, progress } = state;
  // -1 means "the whole day", which is the light at the end of it.
  const clock = progress < 0 ? Math.min(0.999, sun.sunset) : progress;
  const light = daylight(clock, sun);

  // Growth is computed once and shared: the soil shadows have to agree
  // with the plants, or a garden mid-morning is a row of shadows cast by
  // species that have not sung yet.
  const grown = state.placed.map((p) => growthOf(p.plant, progress));

  drawSky(ctx, w, h, theme, light);
  drawGround(ctx, w, h, theme, light, state.placed, grown);

  state.placed.forEach((p, i) => {
    if (grown[i] > 0) drawPlant(ctx, w, h, p, grown[i], state);
  });

  if (!state.still) drawMotes(ctx, w, h, theme, light, state.motes, state.time);
  drawPollinators(ctx, w, h, state);
  drawScale(ctx, w, h, theme, light);
}

/**
 * How grown a plant is at a given point in the day, in [0, 1].
 *
 * Growth is the share of the species' calls that had happened by now, so a
 * bird that sang at dawn and fell silent is fully grown by breakfast and a
 * bird that calls all day is still filling in at dusk. That is the whole
 * argument for scrubbing the day rather than showing a still: the garden
 * grows in the order the yard actually woke up.
 */
function growthOf(plant: BiomePlant, progress: number): number {
  if (progress < 0) return 1;
  const hour = progress * 24;
  const whole = Math.floor(hour);
  let done = 0;
  for (let i = 0; i < whole && i < 24; i++) done += plant.hours[i];
  if (whole < 24) done += plant.hours[whole] * (hour - whole);
  const total = plant.calls || 1;
  // A seedling shows the moment the first call lands, rather than fading
  // up from nothing for the first quarter hour.
  const share = done / total;
  return share <= 0 ? 0 : Math.min(1, 0.22 + 0.78 * share);
}

function drawSky(ctx: CanvasRenderingContext2D, w: number, h: number, theme: Theme, light: number) {
  const top = mix(theme.panel, theme.isDark ? [12, 16, 26] : [206, 220, 232], 0.55 - 0.45 * light);
  const horizon = mix(
    theme.surface,
    theme.isDark ? [26, 30, 38] : [236, 234, 214],
    0.6 - 0.5 * light
  );
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, rgb(top));
  g.addColorStop(1, rgb(horizon));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
}

function drawGround(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  theme: Theme,
  light: number,
  placed: Placed[],
  grown: number[]
) {
  // The bed is a soft wash rather than a line, so the plants look bedded
  // in rather than stood on a shelf.
  const soil = mix(theme.paper, theme.isDark ? [18, 24, 20] : [120, 112, 88], 0.45 - 0.2 * light);
  const g = ctx.createLinearGradient(0, BED_TOP * h - h * 0.06, 0, h);
  g.addColorStop(0, rgba(soil, 0));
  g.addColorStop(0.45, rgba(soil, 0.55));
  g.addColorStop(1, rgba(soil, 0.92));
  ctx.fillStyle = g;
  ctx.fillRect(0, BED_TOP * h - h * 0.06, w, h);

  // A shadow pooled under each base, which is what actually sells the
  // depth the bed is sorted by.
  placed.forEach((p, i) => {
    const growth = grown[i];
    if (growth <= 0) return;
    const r = p.height * h * 0.32 * growth;
    const shade = ctx.createRadialGradient(p.x * w, p.y * h, 0, p.x * w, p.y * h, r);
    shade.addColorStop(0, rgba(theme.ink, 0.2 - 0.1 * p.depth));
    shade.addColorStop(1, rgba(theme.ink, 0));
    ctx.fillStyle = shade;
    ctx.beginPath();
    ctx.ellipse(p.x * w, p.y * h, r, r * 0.22, 0, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawPlant(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  p: Placed,
  growth: number,
  state: SceneState
) {
  const lit = state.hovered === p.plant.species || state.picked === p.plant.species;
  const key = `${Math.round(h)}:${lit ? 1 : 0}:${p.plant.flowering ? 1 : 0}`;
  const stale =
    !p.sprite ||
    p.spriteKey !== key ||
    Math.abs(growth - p.spriteGrowth) > GROWTH_STEP ||
    (growth === 1 && p.spriteGrowth !== 1);

  if (stale) {
    renderSprite(p, growth, h, state.theme, lit);
    p.spriteGrowth = growth;
    p.spriteKey = key;
  }
  if (!p.sprite) return;

  // The sway is applied to the finished sprite rather than baked into it,
  // which is what keeps a still image cheap and a breeze free. Tall thin
  // plants move more than a moss cushion, as they should.
  const sway = state.still
    ? 0
    : Math.sin(state.time * (0.5 + p.depth * 0.25) + p.plant.seed % 10) *
      0.016 *
      p.plant.reach;

  const px = p.x * w;
  const py = p.y * h;
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(sway);
  ctx.drawImage(p.sprite, -p.sprite.width / 2, -p.sprite.height + p.sprite.height * SPRITE_FOOT);
  ctx.restore();
}

/**
 * How much of the sprite sits below the plant's base, as a fraction of its
 * height. Branches that dip under the base need room in the bitmap or they
 * get clipped off at the soil line.
 */
const SPRITE_FOOT = 0.08;

function renderSprite(p: Placed, growth: number, canvasH: number, theme: Theme, lit: boolean) {
  const heightPx = p.height * canvasH;
  const spread = Math.max(0.35, Math.max(-p.skeleton.left, p.skeleton.right) * 2 + 0.2);
  const w = Math.ceil(heightPx * spread);
  const h = Math.ceil(heightPx * (1 + SPRITE_FOOT * 2));
  if (w < 2 || h < 2) return;

  const off = p.sprite && p.sprite.width === w && p.sprite.height === h
    ? p.sprite
    : document.createElement("canvas");
  off.width = w;
  off.height = h;
  const ctx = off.getContext("2d");
  if (!ctx) return;

  const baseX = w / 2;
  const baseY = h * (1 - SPRITE_FOOT);
  const toX = (v: number) => baseX + v * heightPx;
  const toY = (v: number) => baseY + v * heightPx;

  const stem = hsl(
    p.plant.foliage.hue,
    p.plant.foliage.sat,
    p.plant.foliage.light * (theme.isDark ? 1.15 : 0.95)
  );
  const leaf = hsl(
    p.plant.foliage.hue + 6,
    p.plant.foliage.sat + 0.1,
    Math.min(0.72, p.plant.foliage.light + 0.14)
  );
  const bloom = hsl(p.plant.bloom.hue, p.plant.bloom.sat, p.plant.bloom.light);
  const bloomCore = hsl(
    p.plant.bloom.hue - 8,
    p.plant.bloom.sat + 0.12,
    Math.min(0.86, p.plant.bloom.light + 0.24)
  );

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (lit) {
    ctx.shadowColor = rgba(theme.leaf, 0.55);
    ctx.shadowBlur = Math.max(4, heightPx * 0.05);
  }

  // Stems are stroked in width buckets: changing lineWidth flushes the
  // path, so grouping cuts a few hundred strokes down to about six.
  const baseWidth = Math.max(0.6, heightPx * 0.026 * p.plant.girth);
  const buckets = new Map<number, Segment[]>();
  for (const seg of p.skeleton.segments) {
    if (seg.d0 > growth) continue;
    const px = Math.max(0.5, seg.w * baseWidth);
    const bucket = Math.round(px * 2) / 2;
    const list = buckets.get(bucket);
    if (list) list.push(seg);
    else buckets.set(bucket, [seg]);
  }

  ctx.strokeStyle = stem;
  for (const [width, segs] of buckets) {
    ctx.lineWidth = width;
    ctx.beginPath();
    for (const seg of segs) {
      // A segment the growth front is halfway through is drawn halfway.
      const f = seg.d1 <= growth ? 1 : (growth - seg.d0) / Math.max(1e-6, seg.d1 - seg.d0);
      ctx.moveTo(toX(seg.x1), toY(seg.y1));
      ctx.lineTo(
        toX(seg.x1 + (seg.x2 - seg.x1) * f),
        toY(seg.y1 + (seg.y2 - seg.y1) * f)
      );
    }
    ctx.stroke();
  }

  const leafLen = heightPx * 0.055 * (0.7 + 0.5 * p.plant.energy);
  ctx.fillStyle = leaf;
  for (const o of p.skeleton.ornaments) {
    if (o.kind !== "leaf" || o.d > growth) continue;
    // Leaves unfurl over the last stretch of their own growth rather than
    // popping in at full size.
    const f = Math.min(1, (growth - o.d) * 14 + 0.25);
    const len = leafLen * f * (0.7 + o.wobble * 0.6);
    ctx.save();
    ctx.translate(toX(o.x), toY(o.y));
    ctx.rotate(o.angle + (o.wobble - 0.5) * 0.9);
    ctx.beginPath();
    ctx.ellipse(len * 0.5, 0, len * 0.5, len * 0.22, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  const petals = p.plant.bloom.petals;
  const bloomR = heightPx * 0.045 * p.plant.bloom.size;
  for (const o of p.skeleton.ornaments) {
    if (o.kind !== "bloom" || o.d > growth) continue;
    const f = Math.min(1, (growth - o.d) * 10 + 0.2);
    const r = bloomR * f * (0.8 + o.wobble * 0.4);
    ctx.save();
    ctx.translate(toX(o.x), toY(o.y));
    ctx.rotate(o.wobble * Math.PI);
    // A pollinator has been to this plant, so the blooms are open and
    // rimmed in gold. That is the design's "full flowering state".
    if (p.plant.flowering) {
      ctx.shadowColor = "rgba(214, 170, 66, 0.75)";
      ctx.shadowBlur = r * (p.plant.confirmed ? 3 : 1.5);
    }
    ctx.fillStyle = bloom;
    for (let i = 0; i < petals; i++) {
      const a = (i / petals) * Math.PI * 2;
      ctx.beginPath();
      ctx.ellipse(Math.cos(a) * r * 0.6, Math.sin(a) * r * 0.6, r * 0.55, r * 0.34, a, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
    ctx.fillStyle = bloomCore;
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.32, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  p.sprite = off;
}

/**
 * Pollinators: a camera sighting of a species the microphone also heard.
 *
 * A confirmed one, seen and heard inside the correlation window, is bright
 * gold and trails light. A same-day co-occurrence is smaller and paler,
 * because it is a weaker claim and should not look like the same event.
 */
function drawPollinators(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  state: SceneState
) {
  const byName = new Map(state.placed.map((p) => [p.plant.species, p]));
  for (const v of state.pollinators) {
    const home = byName.get(v.species);
    if (!home) continue;
    if (state.progress >= 0 && state.progress < v.time_of_day) continue;

    const phase = state.time * (v.confirmed ? 0.9 : 0.6) + v.detection_id;
    const orbit = home.height * h * 0.42;
    const x = home.x * w + Math.cos(phase) * orbit * 0.8;
    const y = (home.y - home.height * 0.72) * h + Math.sin(phase * 1.7) * orbit * 0.35;
    const size = Math.max(1.6, home.height * h * (v.confirmed ? 0.045 : 0.03));

    const glow = ctx.createRadialGradient(x, y, 0, x, y, size * 4);
    glow.addColorStop(0, v.confirmed ? "rgba(240, 205, 110, 0.85)" : "rgba(214, 200, 160, 0.4)");
    glow.addColorStop(1, "rgba(240, 205, 110, 0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, size * 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = v.confirmed ? "#f6e2a6" : "#cfc4a4";
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();

    // Wings, two short arcs beating faster than the orbit turns.
    const beat = Math.sin(state.time * 18 + v.detection_id) * 0.5 + 0.6;
    ctx.strokeStyle = v.confirmed ? "rgba(246, 226, 166, 0.7)" : "rgba(207, 196, 164, 0.45)";
    ctx.lineWidth = Math.max(0.6, size * 0.35);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - size * 2.2, y - size * 1.6 * beat);
    ctx.moveTo(x, y);
    ctx.lineTo(x + size * 2.2, y - size * 1.6 * beat);
    ctx.stroke();
  }
}

function drawMotes(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  theme: Theme,
  light: number,
  motes: Mote[],
  time: number
) {
  ctx.fillStyle = rgba(theme.isDark ? [220, 230, 210] : [255, 252, 240], 0.16 + 0.1 * (1 - light));
  for (const m of motes) {
    m.x += m.vx + Math.sin(time * 0.6 + m.y) * 0.12;
    m.y += m.vy;
    if (m.y < -8) {
      m.y = h + 8;
      m.x = Math.random() * w;
    }
    if (m.x < -8) m.x = w + 8;
    if (m.x > w + 8) m.x = -8;
    ctx.beginPath();
    ctx.arc(m.x, m.y, m.size, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * The frequency ruler along the soil line.
 *
 * The garden's x axis is pitch, so it is worth labelling: without this the
 * ordering reads as arbitrary, and with it the bed reads as the bottom of
 * a spectrogram.
 */
function drawScale(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  theme: Theme,
  light: number
) {
  const y = h * 0.972;
  ctx.strokeStyle = rgba(theme.muted, 0.3);
  ctx.lineWidth = Math.max(0.6, h * 0.0012);
  ctx.beginPath();
  ctx.moveTo(w * 0.05, y);
  ctx.lineTo(w * 0.95, y);
  ctx.stroke();

  ctx.fillStyle = rgba(theme.muted, 0.75 + 0.2 * (1 - light));
  ctx.font = `${Math.max(8, Math.round(h * 0.021))}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const hz of HZ_TICKS) {
    const t =
      (Math.log2(hz) - Math.log2(HZ_FLOOR)) / (Math.log2(HZ_CEILING) - Math.log2(HZ_FLOOR));
    const x = (0.06 + t * 0.88) * w;
    ctx.beginPath();
    ctx.moveTo(x, y - h * 0.008);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.fillText(hz >= 1000 ? `${hz / 1000}k` : `${hz}`, x, y + h * 0.006);
  }
  ctx.textAlign = "left";
  ctx.fillText("Hz", w * 0.955, y + h * 0.006);
}

// ── Small helpers ───────────────────────────────────────────────────────

/**
 * How lit the yard is at a point in the day, 0 at night and 1 at noon.
 * Ramps over the hour either side of sunrise and sunset, which is roughly
 * how long the yard actually takes to change.
 */
function daylight(clock: number, sun: BiomeGarden["sun"]): number {
  const edge = 1 / 24;
  if (clock < sun.sunrise - edge || clock > sun.sunset + edge) return 0;
  if (clock < sun.sunrise + edge) return (clock - (sun.sunrise - edge)) / (edge * 2);
  if (clock > sun.sunset - edge) return ((sun.sunset + edge) - clock) / (edge * 2);
  return 1;
}

function makeMotes(w: number, h: number): Mote[] {
  return Array.from({ length: MOTE_COUNT }, () => ({
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * 0.25,
    vy: -0.1 - Math.random() * 0.35,
    size: 0.6 + Math.random() * 1.4,
  }));
}

function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 2246822519);
  h = (h ^ (h >>> 13)) * 1274126177;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function rgb(c: RGB): string {
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

function rgba(c: RGB, a: number): string {
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
}

function mix(a: RGB, b: RGB, t: number): RGB {
  const k = Math.max(0, Math.min(1, t));
  return [
    Math.round(a[0] + (b[0] - a[0]) * k),
    Math.round(a[1] + (b[1] - a[1]) * k),
    Math.round(a[2] + (b[2] - a[2]) * k),
  ];
}

function hsl(hue: number, sat: number, light: number): string {
  return `hsl(${((hue % 360) + 360) % 360}deg ${Math.round(
    Math.max(0, Math.min(1, sat)) * 100
  )}% ${Math.round(Math.max(0, Math.min(1, light)) * 100)}%)`;
}

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

function readTheme(): Theme {
  if (typeof document === "undefined") return FALLBACK_THEME;
  const css = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: RGB): RGB => {
    const parts = css.getPropertyValue(name).trim().split(/[\s,]+/).map(Number);
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
