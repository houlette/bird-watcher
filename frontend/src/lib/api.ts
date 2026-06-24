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

export async function fetchDetections(params: {
  limit?: number;
  species_id?: number;
  species_name?: string;
  before?: string;
  only_not_a_bird?: boolean;
  only_unidentified?: boolean;
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
  detections_scene_mask_suppressed: number;
  yolo_bird_rate: number | null;
  classifier_label_rate: number | null;
  user_fp_rate: number | null;
  classifier_accuracy: number | null;
  payload: {
    hour_of_day?: number[];
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
