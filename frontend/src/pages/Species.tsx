import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import DetectionCard from "../components/DetectionCard";
import { fetchDetections } from "../lib/api";

const PLATE_LIMIT = 100;

export default function Species() {
  const { id } = useParams();
  const speciesId = Number(id);
  const { data, isLoading } = useQuery({
    queryKey: ["detections", "species", speciesId],
    queryFn: () => fetchDetections({ species_id: speciesId, limit: PLATE_LIMIT }),
  });

  if (isLoading) return <p className="text-muted mt-4">Loading…</p>;
  if (!data?.length)
    return (
      <p className="font-serif italic text-lg text-muted mt-6">
        No detections for this species yet.
      </p>
    );

  // The fetch is capped at PLATE_LIMIT, so a full page means "at least
  // this many" — don't claim an exact total we don't have.
  const countLabel =
    data.length >= PLATE_LIMIT
      ? `${PLATE_LIMIT}+ sightings (most recent ${PLATE_LIMIT} shown)`
      : `${data.length} sighting${data.length === 1 ? "" : "s"}`;

  return (
    <div>
      <div className="mb-4">
        <div className="fg-overline">Species plate</div>
        <h2 className="font-serif font-medium text-2xl text-ink leading-tight mt-0.5">
          {data[0].species}
        </h2>
        {data[0].scientific_name && (
          <div className="font-serif italic text-muted mt-0.5">
            {data[0].scientific_name}
          </div>
        )}
        <div className="text-sm text-faint mt-1 tnum">{countLabel}</div>
        <Link
          to="/"
          className="inline-block text-sm text-muted hover:text-leaf underline underline-offset-2 mt-1.5 transition-colors"
        >
          ← Back to feed
        </Link>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {data.map((d) => (
          <DetectionCard key={d.id} detection={d} compact />
        ))}
      </div>
    </div>
  );
}
