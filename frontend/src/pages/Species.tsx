import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import DetectionCard from "../components/DetectionCard";
import { fetchDetections } from "../lib/api";

export default function Species() {
  const { id } = useParams();
  const speciesId = Number(id);
  const { data, isLoading } = useQuery({
    queryKey: ["detections", "species", speciesId],
    queryFn: () => fetchDetections({ species_id: speciesId, limit: 100 }),
  });

  if (isLoading) return <p className="text-muted mt-4">Loading…</p>;
  if (!data?.length)
    return (
      <p className="font-serif italic text-lg text-muted mt-6">
        No detections for this species yet.
      </p>
    );

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
        <div className="text-sm text-faint mt-1 tnum">{data.length} sightings</div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {data.map((d) => (
          <DetectionCard key={d.id} detection={d} compact />
        ))}
      </div>
    </div>
  );
}
