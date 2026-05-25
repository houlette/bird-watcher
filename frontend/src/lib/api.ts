export type Detection = {
  id: number;
  visit_id: number;
  species: string | null;
  scientific_name: string | null;
  confidence: number;
  audio_confirmed: boolean;
  raw_predictions: { species: string; p: number }[];
  crop_url: string;
  bbox: [number, number, number, number];
  track_id: number;
  created_at: string;
};

export async function fetchDetections(params: { limit?: number; species_id?: number } = {}) {
  const url = new URL("/api/detections", window.location.origin);
  if (params.limit) url.searchParams.set("limit", String(params.limit));
  if (params.species_id) url.searchParams.set("species_id", String(params.species_id));
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetchDetections: ${r.status}`);
  return (await r.json()) as Detection[];
}

export type SpeciesEntry = { name: string; total: number };
export type SpeciesList = {
  source: "calibration" | "fallback";
  // Yard-known species (Haikubox-heard) sorted by detection count desc.
  yard: SpeciesEntry[];
  // Broader NA-bird list (alphabetical) — pigeons, raptors, etc. the
  // user might see visually without the Haikubox having heard them.
  extra: SpeciesEntry[];
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
  return (await r.json()) as { ok: boolean; species_id: number; species: string };
}
