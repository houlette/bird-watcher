import { useQuery } from "@tanstack/react-query";

import DetectionCard from "../components/DetectionCard";
import { fetchDetections } from "../lib/api";

export default function Feed() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["detections"],
    queryFn: () => fetchDetections({ limit: 50 }),
    refetchInterval: 15_000,
  });

  if (isLoading) return <p className="text-slate-500">Loading…</p>;
  if (error) return <p className="text-red-600">Failed to load detections.</p>;
  if (!data || data.length === 0) {
    return (
      <div className="text-slate-500 text-center py-10">
        <p className="text-lg">No birds yet.</p>
        <p className="text-sm">Once the camera fires a motion event, detections will appear here.</p>
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      {data.map((d) => (
        <DetectionCard key={d.id} detection={d} />
      ))}
    </div>
  );
}
