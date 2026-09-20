export type Detection = {
  id: number;
  visit_id: number;
  // Null when the classifier rejected every crop ("Unidentified").
  // Used to link the card title to the /species/:id plate page.
  species_id: number | null;
  species: string | null;
  scientific_name: string | null;
  confidence: number;
  audio_confirmed: boolean;
  raw_predictions: { species: string; p: number }[];
  crop_url: string;
  bbox: [number, number, number, number];
  track_id: number;
  // When the camera actually recorded the bird (parsed from the Reolink
  // filename). This is what the feed displays.
  captured_at: string;
  // When the worker finished processing the clip. May be hours later than
  // captured_at during a backlog drain. Kept for debugging.
  created_at: string;
  // Opaque cursor — pass back as `before` to get the next (older) page.
  cursor: string;
  // Non-null only when the current Detection.species came from a non-UI
  // source. `"llm-claude"` means scripts/llm_classify_unidentified.py
  // generated this label and we should surface the rationale (below) so
  // the user can spot-check what Claude saw.
  correction_source: string | null;
  correction_rationale: string | null;
  // Wikipedia thumbnail of the species, used for side-by-side
  // comparison in the review-mode cards. Null when we haven't
  // fetched one or Wikipedia has no usable image.
  reference_image_url: string | null;
  // Per-crop quality metrics populated at ingest. NULL on legacy
  // rows that haven't been backfilled. Drives the "bad crops" filter
  // and the small score footer on each card.
  crop_area_px: number | null;
  brightness: number | null;     // 0-255 (mean grayscale)
  sharpness: number | null;      // Laplacian variance
  // P(NAB) the binary post-filter scored; non-null only when it overrode
  // this crop to "Not a bird". Drives the binary-filter audit feed.
  nab_override_p: number | null;
};

export type CropVariantsMeta = {
  detection_id: number;
  has_raw: boolean;
  has_initial: boolean;
  has_sr?: boolean;
  sharpness: number | null;
  crop_area_px: number | null;
  brightness: number | null;
  variants: Record<string, string>;
};

export async function fetchCropVariantsMeta(detectionId: number): Promise<CropVariantsMeta> {
  const r = await fetch(`/api/detections/${detectionId}/crop-variants`);
  if (!r.ok) throw new Error(`fetchCropVariantsMeta: ${r.status}`);
  return (await r.json()) as CropVariantsMeta;
}

export function getCropVariantUrl(
  detectionId: number,
  options: {
    chroma?: boolean;
    clahe?: boolean;
    mertens?: boolean;
    sharpen?: boolean;
    sr?: boolean;
    preset?: string;
    source?: "lucky" | "initial";
  } = {},
): string {
  const params = new URLSearchParams();
  if (options.preset) {
    params.set("preset", options.preset);
  } else {
    if (options.chroma !== undefined) params.set("chroma", options.chroma ? "1" : "0");
    if (options.clahe !== undefined) params.set("clahe", options.clahe ? "1" : "0");
    if (options.mertens !== undefined) params.set("mertens", options.mertens ? "1" : "0");
    if (options.sharpen !== undefined) params.set("sharpen", options.sharpen ? "1" : "0");
    if (options.sr !== undefined) params.set("sr", options.sr ? "1" : "0");
  }
  if (options.source && options.source !== "lucky") {
    params.set("source", options.source);
  }
  const qs = params.toString();
  return `/api/detections/${detectionId}/crop${qs ? `?${qs}` : ""}`;
}

export async function fetchDetections(params: {
  limit?: number;
  species_id?: number;
  species_name?: string;
  before?: string;
  only_not_a_bird?: boolean;
  only_unidentified?: boolean;
  interesting?: boolean;
  awaiting_review?: boolean;
  bad_quality?: boolean;
  binary_nab?: boolean;
} = {}) {
  const url = new URL("/api/detections", window.location.origin);
  if (params.limit) url.searchParams.set("limit", String(params.limit));
  if (params.species_id) url.searchParams.set("species_id", String(params.species_id));
  if (params.species_name) url.searchParams.set("species_name", params.species_name);
  if (params.before) url.searchParams.set("before", params.before);
  if (params.only_not_a_bird) url.searchParams.set("only_not_a_bird", "true");
  if (params.only_unidentified) url.searchParams.set("only_unidentified", "true");
  if (params.interesting) url.searchParams.set("interesting", "true");
  if (params.awaiting_review) url.searchParams.set("awaiting_review", "true");
  if (params.bad_quality) url.searchParams.set("bad_quality", "true");
  if (params.binary_nab) url.searchParams.set("binary_nab", "true");
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetchDetections: ${r.status}`);
  return (await r.json()) as Detection[];
}

export type SpeciesEntry = {
  name: string;
  // Audio-detection count from the Haikubox calibration (familiarity hint).
  total: number;
  // Number of classified crops we have for this species in the DB. The
  // Filter picker sorts its yard list on this.
  crops?: number;
  reference_image_url?: string | null;
};
export type FamilyEntry = {
  name: string;
  members: string[];
  reference_image_url?: string | null;
};
export type SpeciesList = {
  source: "calibration" | "fallback";
  // Yard-known species (Haikubox-heard) sorted by detection count desc.
  yard: SpeciesEntry[];
  // Broader NA-bird list (alphabetical) — pigeons, raptors, etc. the
  // user might see visually without the Haikubox having heard them.
  extra: SpeciesEntry[];
  // Family-level catch-all labels for "I know it's a sparrow but
  // I can't tell which kind." `members` is the constituent species.
  families: FamilyEntry[];
  // Legacy: equals yard. Kept for backward compat.
  species: SpeciesEntry[];
};

export async function fetchSpecies(): Promise<SpeciesList> {
  const r = await fetch("/api/species");
  if (!r.ok) throw new Error(`fetchSpecies: ${r.status}`);
  return (await r.json()) as SpeciesList;
}

export async function submitCorrection(detection_id: number, correct_species_name: string) {
  const r = await fetch("/api/corrections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ detection_id, correct_species_name }),
  });
  if (!r.ok) throw new Error(`submitCorrection: ${r.status}`);
  return (await r.json()) as {
    ok: boolean;
    species_id: number;
    species: string;
    // Identifies the row to delete (and value to restore) on undo.
    correction_id: number;
    prev_species_id: number | null;
  };
}

// Reverse a single just-made correction or confirmation. Backs the "Undo"
// toast: deletes the Correction row the action created and restores the
// detection's pre-action species_id.
export async function undoCorrection(
  correction_id: number,
  restore_species_id: number | null,
) {
  const r = await fetch("/api/corrections/undo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ correction_id, restore_species_id }),
  });
  if (!r.ok) throw new Error(`undoCorrection: ${r.status}`);
  return (await r.json()) as {
    ok: boolean;
    detection_id: number;
    restored_species_id: number | null;
  };
}

// ---- Stats ---------------------------------------------------------------

export type DailyStats = {
  date: string;
  clips_received: number;
  clips_daylight: number;
  clips_with_detections: number;
  detections_total: number;
  detections_labeled_by_classifier: number;
  detections_user_corrected: number;
  corrections_nab: number;
  corrections_unknown: number;
  corrections_real_species: number;
  classifier_correct: number;
  classifier_eligible: number;
  visits_with_processing_error: number;
  detections_audio_confirmed: number;
  // The three spatial defences, counted separately so each can be judged
  // on its own. These are frame-level YOLO boxes, not detections — one
  // detection is a track over many frames, so they don't share a scale
  // with detections_total.
  detections_scene_mask_suppressed: number;
  detections_recurrence_suppressed: number;
  detections_backdrop_suppressed: number;
  // How many boxes the backdrop model actually scored. Without it,
  // "suppressed 0" reads the same whether it rejected nothing or was
  // never loaded for those hours.
  detections_backdrop_scored: number;
  yolo_bird_rate: number | null;
  classifier_label_rate: number | null;
  user_fp_rate: number | null;
  classifier_accuracy: number | null;
  payload: {
    hour_of_day?: number[];
    // Detections the binary bird-or-not head overrode to NAB. Counted as
    // `nab_override_p IS NOT NULL`, which corrections never clear, so the
    // split against detections_total stays exact after review.
    binary_nab_overrides?: number;
    yolo_confidence_hist?: { nab: number[]; species: number[] };
    // Per-day crop-quality percentiles. Each metric is nullable when no
    // detections that day had the column populated (legacy rows
    // pre-backfill, or no detections at all).
    crop_quality?: {
      sharpness: { p25: number; p50: number; p75: number; n: number } | null;
      brightness: { p25: number; p50: number; p75: number; n: number } | null;
      area_px: { p25: number; p50: number; p75: number; n: number } | null;
    };
    [k: string]: unknown;
  };
};

export type TrainingDataEntry = {
  species: string;
  gold: number;
  high: number;
  medium: number;
  total: number;
};

export type StatsResponse = {
  daily: DailyStats[];
  totals: {
    visits_total: number;
    detections_total: number;
    corrections_total: number;
    pending_backlog: number;
    ready_to_fine_tune_species: number;
    top_species: { species: string; count: number }[];
    species_accuracy: { species: string; n: number; accuracy: number }[];
    // Per-species breakdown of correction labels by trust tier. Drives
    // the "Training data progress" card so we can see which species are
    // ready to fine-tune on and how much the MED-review queue is worth.
    training_data: TrainingDataEntry[];
    training_ready_species: number;
    review_queue_size: number;
  };
  as_of: string;
};

export async function fetchStats(days = 30): Promise<StatsResponse> {
  const url = new URL("/api/stats", window.location.origin);
  url.searchParams.set("days", String(days));
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetchStats: ${r.status}`);
  return (await r.json()) as StatsResponse;
}

// ---- Species activity (Insights tab) ------------------------------------

export type SpeciesActivity = {
  species_id: number;
  species: string;
  scientific_name: string | null;
  total: number;
  // ISO timestamps in the feeder's local timezone.
  first_seen: string;
  last_seen: string;
  // 24 buckets, index = local hour 0–23.
  by_hour: number[];
  // 53 buckets, index = (local day-of-year − 1) // 7 — i.e. fixed 7-day
  // weeks tiling Jan 1 forward. Bucket 0 ≈ first week of January.
  by_week: number[];
};

export type ActivityResponse = {
  // IANA timezone the buckets were computed in (the feeder's location).
  tz: string;
  species: SpeciesActivity[];
  as_of: string;
};

export async function fetchSpeciesActivity(): Promise<ActivityResponse> {
  const r = await fetch("/api/stats/activity");
  if (!r.ok) throw new Error(`fetchSpeciesActivity: ${r.status}`);
  return (await r.json()) as ActivityResponse;
}

export async function confirmClassifierLabel(detection_id: number) {
  const r = await fetch(`/api/corrections/confirm/${detection_id}`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(`confirmClassifierLabel: ${r.status}`);
  return (await r.json()) as {
    ok: boolean;
    detection_id: number;
    source: string;
    species: string | null;
    // For the undo toast: confirming leaves species_id unchanged, so
    // prev_species_id echoes the current value.
    correction_id: number;
    prev_species_id: number | null;
  };
}

export async function bulkCorrection(detection_ids: number[], correct_species_name: string) {
  // Abort after 20s (> the backend's 15s SQLite busy_timeout) so a request
  // that stalls on write-lock contention or a dropped connection rejects
  // cleanly instead of leaving the mutation pending forever — which froze
  // the bulk action bar with no way to recover.
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20_000);
  let r: Response;
  try {
    r = await fetch("/api/corrections/bulk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ detection_ids, correct_species_name }),
      signal: ctrl.signal,
    });
  } catch (e) {
    if (ctrl.signal.aborted) throw new Error("bulkCorrection: timed out");
    throw e;
  } finally {
    clearTimeout(timer);
  }
  if (!r.ok) throw new Error(`bulkCorrection: ${r.status}`);
  return (await r.json()) as {
    ok: boolean;
    count: number;
    species: string;
    results: Array<{ id: number; species_id: number; species: string }>;
  };
}

// ── Art page ────────────────────────────────────────────────────────────
// The /api/art endpoints speak in flights. A flight is one DETECTION —
// one track — not one visit: a visit with forty detections is forty
// separate birds, and drawing them as a single path would connect birds
// that never met. Times come back offset-aware (unlike the detections
// feed's naive UTC), already converted to the feeder's local zone, so
// render them as-is without appending a "Z".

export type ArtPoint = {
  /** Centre position and box size as fractions of the 4K frame. */
  x: number;
  y: number;
  w: number;
  h: number;
  /** Position along the path, 0 at the start and 1 at the end. */
  t: number;
};

export type ArtStyle = {
  primary: string;
  accent: string;
  /** Rough centre of the species' vocal range, in hertz. */
  call_hz: number;
  /** Typical adult body mass in grams; drives stroke weight and node size. */
  mass_g: number;
  /** call_hz snapped to a pentatonic scale, so overlapping chimes agree. */
  chime_hz: number;
};

export type ArtFlight = {
  detection_id: number;
  visit_id: number;
  /** ISO with a UTC offset, in the camera's timezone. */
  started_at: string;
  /** Fraction of the local day, 0 at midnight. Drives the timeline. */
  time_of_day: number;
  duration_seconds: number;
  /** "Unidentified" when the classifier rejected every crop. */
  species: string;
  species_id: number | null;
  scientific_name: string | null;
  confidence: number;
  audio_confirmed: boolean;
  crop_url: string | null;
  style: ArtStyle;
  /**
   * "tracked" means the path is the frames the bird was really seen in.
   * "synthesized" means only the perch is real and the approach and
   * departure are invented. Most rows predate per-frame tracking, so the
   * UI has to say which it's showing.
   */
  path_kind: "tracked" | "synthesized";
  points: ArtPoint[];
};

export type ArtSun = {
  /** Sunrise and sunset as fractions of the local day. */
  sunrise: number;
  sunset: number;
};

export type ArtDay = {
  date: string;
  tz: string;
  sun: ArtSun;
  flights: ArtFlight[];
  summary: {
    returned: number;
    total_available: number;
    truncated: boolean;
    tracked: number;
    species_counts: Record<string, number>;
    species_colors: Record<string, string>;
  };
};

export type ArtDateSummary = {
  date: string;
  flight_count: number;
  top_species: { species: string; count: number }[];
};

export async function fetchArtDay(params: {
  date?: string;
  limit?: number;
  species_id?: number;
  include_unidentified?: boolean;
} = {}): Promise<ArtDay> {
  const url = new URL("/api/art/trajectories", window.location.origin);
  if (params.date) url.searchParams.set("date", params.date);
  if (params.limit) url.searchParams.set("limit", String(params.limit));
  if (params.species_id) url.searchParams.set("species_id", String(params.species_id));
  if (params.include_unidentified === false) {
    url.searchParams.set("include_unidentified", "false");
  }
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`fetchArtDay: ${r.status}`);
  return (await r.json()) as ArtDay;
}

export async function fetchArtDates(limit = 60): Promise<{ tz: string; dates: ArtDateSummary[] }> {
  const url = new URL("/api/art/dates", window.location.origin);
  url.searchParams.set("limit", String(limit));
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`fetchArtDates: ${r.status}`);
  return (await r.json()) as { tz: string; dates: ArtDateSummary[] };
}

// ── The Perch & Flagon ──────────────────────────────────────────────────
// The tavern reads the same detections the feed does and seats each one as
// a guest. A patron is one DETECTION, not one visit, for the same reason a
// flight is: a visit with forty detections is forty separate birds.
//
// Times come back offset-aware in the feeder's zone, like the Art page's
// and unlike the feed's naive UTC. Render them as-is.

export type TavernArchetype = {
  /** One of messenger, adventurer, local, minstrel, wanderer, hunter, stranger. */
  id: string;
  title: string;
  role: string;
  /** Region of the room: bar, booth, bench, hearth, rafters, shadow. */
  seat: string;
  blurb: string;
};

export type TavernStyle = {
  primary: string;
  accent: string;
  call_hz: number;
  mass_g: number;
  chime_hz: number;
};

export type TavernPatron = {
  detection_id: number;
  visit_id: number;
  /** Null when the classifier would not name the bird: a cloaked stranger. */
  species: string | null;
  species_id: number | null;
  scientific_name: string | null;
  arrived_at: string;
  duration_seconds: number;
  audio_confirmed: boolean;
  confidence: number;
  crop_url: string | null;
  archetype: TavernArchetype;
  style: TavernStyle;
  rarity: "common" | "regular" | "uncommon" | "rare";
  /** Shillings this guest left on the table. */
  payment: number;
  order: { drink: string; dish: string };
  line: string;
  /** How long they linger in the room, relative to the other guests. */
  dwell: number;
  /**
   * True when the guest is shown only because the room would otherwise be
   * empty. They are not here now; they were the last company the camera saw.
   */
  stale: boolean;
};

export type TavernEvent = {
  detection_id: number;
  visit_id: number;
  at: string;
  id: string;
  title: string;
  line: string;
  crop_url: string | null;
};

export type TavernLedger = {
  /** Spendable: the founding purse plus everything taken since opening. */
  earned: number;
  /** What the house inherited, capped at purse_cap. */
  founding_purse: number;
  purse_cap: number;
  /** Everything the birds ever paid, including the uncapped inheritance. */
  earned_all_time: number;
  since_opening: number;
  spent: number;
  balance: number;
  patrons_served: number;
  named_patrons: number;
};

export type TavernUpgrade = {
  id: string;
  name: string;
  category: string;
  cost: number;
  /** The flag TavernCanvas reads to draw this. */
  effect: string;
  blurb: string;
  owned: boolean;
  affordable: boolean;
};

export type TavernState = {
  tz: string;
  now: string;
  hearth: {
    phase: "dawn" | "day" | "dusk" | "night";
    sunrise_hour: number;
    sunset_hour: number;
    local_hour: number;
  };
  /** No arrivals inside the window; the patrons below are the last company. */
  quiet: boolean;
  window_minutes: number;
  patrons: TavernPatron[];
  events: TavernEvent[];
  ledger: TavernLedger;
  upgrades: TavernUpgrade[];
  unlocked: string[];
  today: { arrivals: number; species: number; top: { species: string; count: number }[] };
  /** Highest detection id in the database; poll /arrivals with it. */
  cursor: number;
  last_seen_detection_id: number | null;
  /** Guests who arrived since the marker /seen keeps: new to this reader. */
  since_last_seen: number;
  opened_at: string | null;
};

export type TavernArrivals = {
  cursor: number;
  patrons: TavernPatron[];
  events: TavernEvent[];
  /** What this batch alone put in the till. */
  earned: number;
};

export type TavernGuestbookEntry = {
  species: string;
  species_id: number;
  scientific_name: string;
  visits: number;
  first_seen: string | null;
  last_seen: string | null;
  times_heard: number;
  rarity: TavernPatron["rarity"];
  style: TavernStyle;
  archetype: TavernArchetype;
  /** Null until the guest ledger upgrade is bought. */
  lore: { title: string; role: string; backstory: string } | null;
};

export async function fetchTavernState(params: {
  window_minutes?: number;
  limit?: number;
  strangers?: boolean;
} = {}): Promise<TavernState> {
  const url = new URL("/api/tavern/state", window.location.origin);
  if (params.window_minutes) url.searchParams.set("window_minutes", String(params.window_minutes));
  if (params.limit) url.searchParams.set("limit", String(params.limit));
  if (params.strangers === false) url.searchParams.set("strangers", "false");
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`fetchTavernState: ${r.status}`);
  return (await r.json()) as TavernState;
}

export async function fetchTavernArrivals(
  after: number,
  strangers = true
): Promise<TavernArrivals> {
  const url = new URL("/api/tavern/arrivals", window.location.origin);
  url.searchParams.set("after", String(after));
  if (!strangers) url.searchParams.set("strangers", "false");
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`fetchTavernArrivals: ${r.status}`);
  return (await r.json()) as TavernArrivals;
}

export async function fetchTavernGuestbook(): Promise<{
  has_ledger: boolean;
  entries: TavernGuestbookEntry[];
}> {
  const r = await fetch("/api/tavern/guestbook");
  if (!r.ok) throw new Error(`fetchTavernGuestbook: ${r.status}`);
  return await r.json();
}

export async function buyTavernUpgrade(upgrade_id: string): Promise<{
  unlocked: string[];
  ledger: TavernLedger;
  upgrades: TavernUpgrade[];
  bought: TavernUpgrade;
}> {
  const r = await fetch("/api/tavern/unlock", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ upgrade_id }),
  });
  if (!r.ok) {
    // 402 carries "N more shillings needed", which is worth showing.
    const detail = await r.json().catch(() => null);
    throw new Error(detail?.detail ?? `buyTavernUpgrade: ${r.status}`);
  }
  return await r.json();
}

export async function markTavernSeen(detection_id: number): Promise<void> {
  await fetch("/api/tavern/seen", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ detection_id }),
  });
}

// ── Biome page ──────────────────────────────────────────────────────────
// The /api/biome endpoints speak in plants, and a plant is a SPECIES, not
// a detection. That is the opposite of the Art and Tavern pages, and it is
// deliberate: the Haikubox does not track individual birds, it reports
// that a species was audible in a window, so seven hundred House Sparrow
// rows are one hedge full of sparrows rather than seven hundred arrivals.
// Times come back offset-aware in the feeder's zone, like the Art page's.

export type BiomeBloom = {
  /** Degrees on the colour wheel, deep indigo at low pitch to rose at high. */
  hue: number;
  sat: number;
  light: number;
  petals: number;
  size: number;
};

export type BiomeFoliage = {
  hue: number;
  sat: number;
  light: number;
};

export type BiomePlant = {
  species: string;
  /** Which L-system the canvas expands: moss, spire, vine, frond or floret. */
  form: "moss" | "spire" | "vine" | "frond" | "floret";
  form_title: string;
  form_blurb: string;
  /** Turtle-graphics L-system. F draws, X buds, +- turn, [] branch, L leaf, O bloom. */
  axiom: string;
  rules: Record<string, string>;
  /** Rewrite passes. Capped server-side so one plant cannot stall the tab. */
  depth: number;
  /** Turn angle in degrees. */
  angle: number;
  /** Stem thickness multiplier, from body mass and vitality. */
  girth: number;
  /** Segment length multiplier, from vitality. */
  reach: number;
  /** How alive the plant looks, 0 to 1. */
  energy: number;
  /**
   * Which measurement produced `energy`. "calls" means call density and
   * persistence only, which is all the v2 REST feed gives: every
   * confidence in the cache is null. "calls+score" folds in BirdNET
   * scores, "spectral" would be real Haikubox specSum energy.
   */
  energy_source: "calls" | "calls+score" | "spectral";
  /** Position on the log-frequency ramp, 0 at 400 Hz and 1 at 8 kHz. */
  pitch: number;
  call_hz: number;
  mass_g: number;
  /** call_hz snapped to a pentatonic scale, so the chorus agrees with itself. */
  chime_hz: number;
  bloom: BiomeBloom;
  foliage: BiomeFoliage;
  plumage: { primary: string; accent: string };
  seed: number;
  calls: number;
  /** This species' share of the day's calls. */
  share: number;
  /** Calls per local hour, 24 entries. */
  hours: number[];
  /** First and last call as fractions of the local day. */
  first_heard: number;
  last_heard: number;
  first_heard_at: string;
  last_heard_at: string;
  mean_confidence: number | null;
  /** The camera logged this species the same day, so a pollinator visits. */
  flowering: boolean;
  /** Seen and heard within the correlation window, the strict claim. */
  confirmed: boolean;
};

export type BiomePollinator = {
  detection_id: number;
  species: string;
  species_id: number | null;
  at: string;
  time_of_day: number;
  /**
   * True means the pipeline matched this sighting to a call inside the
   * correlation window. False means only that the camera saw the species
   * somewhere in the same local day, which the page draws paler.
   */
  confirmed: boolean;
  crop_url: string | null;
  color: string;
};

export type BiomeGarden = {
  date: string;
  tz: string;
  sun: { sunrise: number; sunset: number };
  /** Ordered low pitch to high, so the garden plants itself front to back. */
  plants: BiomePlant[];
  pollinators: BiomePollinator[];
  /** The whole yard's calls per local hour, 24 entries. */
  chorus: number[];
  summary: {
    returned: number;
    species_heard: number;
    calls: number;
    busiest: number;
    peak_hour: number | null;
    truncated: boolean;
    quiet: boolean;
    energy_source: BiomePlant["energy_source"] | null;
  };
};

export type BiomeDateSummary = {
  date: string;
  call_count: number;
  species_count: number;
  top_species: { species: string; count: number }[];
};

export async function fetchBiomeGarden(
  params: { date?: string; limit?: number; min_calls?: number } = {}
): Promise<BiomeGarden> {
  const url = new URL("/api/biome/garden", window.location.origin);
  if (params.date) url.searchParams.set("date", params.date);
  if (params.limit) url.searchParams.set("limit", String(params.limit));
  if (params.min_calls) url.searchParams.set("min_calls", String(params.min_calls));
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`fetchBiomeGarden: ${r.status}`);
  return (await r.json()) as BiomeGarden;
}

export async function fetchBiomeDates(
  limit = 60
): Promise<{ tz: string; dates: BiomeDateSummary[] }> {
  const url = new URL("/api/biome/dates", window.location.origin);
  url.searchParams.set("limit", String(limit));
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`fetchBiomeDates: ${r.status}`);
  return (await r.json()) as { tz: string; dates: BiomeDateSummary[] };
}

// ── Territory page ──────────────────────────────────────────────────────
// /api/territory scores a day of footage as a turf war. Two distinctions
// the UI has to preserve, because both are places this page could easily
// start overstating what the camera saw:
//
//   - `control_basis` says whether a zone's holder was decided on seconds
//     of dwell or on plain appearances. Only tracks carrying
//     `track_frames` have a dwell, so an archive day falls back to
//     counting landings.
//   - A displacement needs two tracks on a shared clock. `summary.timed`
//     and `summary.eligible_visits` say how much of the day could be
//     judged at all; the rest can only be scored for occupancy.

export type TerritoryControl = {
  faction: string;
  name: string;
  color: string;
  seconds: number;
  holds: number;
  /** Share of the zone on whatever `control_basis` says. */
  share: number;
};

export type TerritoryZone = {
  id: string;
  name: string;
  blurb: string;
  /** Centre as a fraction of the frame, x by width and y by height. */
  x: number;
  y: number;
  /** Radius in normalised units, so it draws as a wide ellipse on a 16:9 frame. */
  radius: number;
  seconds: number;
  /** Detections placed in this zone, unclaimed ones included. */
  visits: number;
  /** Birds here the classifier could not name; they hold ground for nobody. */
  unclaimed: number;
  control: TerritoryControl[];
  holder: string | null;
  contested: boolean;
};

export type TerritoryStanding = {
  faction: string;
  name: string;
  style: string;
  blurb: string;
  color: string;
  weight: number;
  seconds: number;
  holds: number;
  zones_held: number;
  species: { species: string; count: number; color: string }[];
  wins: number;
  losses: number;
};

export type TerritoryDisplacement = {
  zone: string;
  zone_name: string;
  visit_id: number;
  winner_detection_id: number | null;
  winner: string | null;
  winner_faction: string | null;
  winner_faction_name: string | null;
  winner_color: string;
  loser_detection_id: number | null;
  loser: string | null;
  loser_faction: string | null;
  loser_faction_name: string | null;
  loser_color: string;
  at_frame: number;
  /** Seconds into the clip the perch changed hands. */
  at_seconds: number;
  /** How long the loser had held it before that. */
  held_seconds: number;
};

export type TerritoryDay = {
  date: string;
  tz: string;
  sun: { sunrise: number; sunset: number };
  /** "built-in" means the zone map is the hard-coded one, not calibrated here. */
  zone_source: "built-in" | "calibrated";
  control_basis: "seconds" | "appearances";
  zones: TerritoryZone[];
  standings: TerritoryStanding[];
  displacements: TerritoryDisplacement[];
  /** The day's scoreboard as a sentence, built from the tallies. */
  dispatch: string;
  summary: {
    detections: number;
    visits: number;
    in_a_zone: number;
    unclaimed: number;
    tracked: number;
    timed: number;
    contested_visits: number;
    eligible_visits: number;
    dwell_quality: Partial<Record<"measured" | "estimated" | "unknown", number>>;
    quiet: boolean;
  };
};

export type TerritoryDateSummary = {
  date: string;
  detection_count: number;
  timed_count: number;
};

export async function fetchTerritoryDay(
  params: { date?: string } = {}
): Promise<TerritoryDay> {
  const url = new URL("/api/territory/day", window.location.origin);
  if (params.date) url.searchParams.set("date", params.date);
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`fetchTerritoryDay: ${r.status}`);
  return (await r.json()) as TerritoryDay;
}

export async function fetchTerritoryDates(
  limit = 60
): Promise<{ tz: string; dates: TerritoryDateSummary[] }> {
  const url = new URL("/api/territory/dates", window.location.origin);
  url.searchParams.set("limit", String(limit));
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`fetchTerritoryDates: ${r.status}`);
  return (await r.json()) as { tz: string; dates: TerritoryDateSummary[] };
}
