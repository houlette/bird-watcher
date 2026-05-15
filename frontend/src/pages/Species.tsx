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

  if (isLoading) return <p>Loading…</p>;
  if (!data?.length) return <p>No detections for this species yet.</p>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-3">{data[0].species}</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {data.map((d) => (
          <DetectionCard key={d.id} detection={d} compact />
        ))}
      </div>
    </div>
  );
}
