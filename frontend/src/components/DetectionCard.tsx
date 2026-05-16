import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { submitCorrection, type Detection } from "../lib/api";
import AudioBadge from "./AudioBadge";
import SpeciesPicker from "./SpeciesPicker";

export default function DetectionCard({
  detection,
  compact = false,
}: {
  detection: Detection;
  compact?: boolean;
}) {
  const time = new Date(detection.created_at).toLocaleString();
  const [pickerOpen, setPickerOpen] = useState(false);
  const queryClient = useQueryClient();

  const correctionMutation = useMutation({
    mutationFn: (species: string) => submitCorrection(detection.id, species),
    onSuccess: () => {
      // Optimistic refresh of the feed — the backend updates Detection.species_id
      // in place, so the corrected label appears on the next refetch.
      queryClient.invalidateQueries({ queryKey: ["detections"] });
      setPickerOpen(false);
    },
  });

  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden flex flex-col">
      <img
        src={detection.crop_url}
        alt={detection.species ?? "bird"}
        className="w-full object-cover aspect-[4/3]"
        loading="lazy"
      />
      <div className={`p-2 ${compact ? "text-xs" : "text-sm"}`}>
        <div className="flex items-center justify-between gap-2">
          <span className="font-semibold">{detection.species ?? "Unidentified"}</span>
          {detection.audio_confirmed && <AudioBadge />}
        </div>
        <div className="text-slate-500">{Math.round(detection.confidence * 100)}% · {time}</div>

        {!compact && (
          <button
            className="mt-2 text-xs text-slate-500 hover:text-forest underline"
            onClick={() => setPickerOpen(true)}
            disabled={correctionMutation.isPending}
          >
            {correctionMutation.isPending ? "Saving…" : "Wrong species?"}
          </button>
        )}
        {correctionMutation.isError && (
          <p className="text-xs text-red-600 mt-1">
            Couldn't save correction: {(correctionMutation.error as Error).message}
          </p>
        )}
      </div>

      <SpeciesPicker
        open={pickerOpen}
        current={detection.species}
        onSelect={(name) => correctionMutation.mutate(name)}
        onCancel={() => setPickerOpen(false)}
      />
    </div>
  );
}
