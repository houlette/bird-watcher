import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { submitCorrection, type Detection } from "../lib/api";
import AudioBadge from "./AudioBadge";
import SpeciesPicker, { NOT_A_BIRD } from "./SpeciesPicker";

export default function DetectionCard({
  detection,
  compact = false,
}: {
  detection: Detection;
  compact?: boolean;
}) {
  // Show CAPTURE time (when the camera saw the bird), not the row's
  // processing time. The API tags captured_at as naive UTC; JS's Date
  // parser interprets a naive ISO string as local time, so append 'Z'
  // before parsing to treat it as UTC and let toLocaleString convert
  // to the viewer's local zone.
  const time = new Date(detection.captured_at + "Z").toLocaleString();
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
          <div className="mt-2 flex items-center justify-between gap-2 text-xs">
            <button
              className="text-slate-500 hover:text-forest underline disabled:opacity-50"
              onClick={() => setPickerOpen(true)}
              disabled={correctionMutation.isPending}
            >
              {correctionMutation.isPending ? "Saving…" : "Wrong species?"}
            </button>
            {/* One-click false-positive label. Most "Unidentified" crops are
                YOLO false positives (wind-stirred leaves, sun glints on the
                feeder), and labeling them via the picker took 2 clicks plus
                a scroll. This shortcut makes bulk labeling tractable; the
                Detection is filtered out of the feed immediately afterward,
                giving the user instant visual confirmation. */}
            <button
              className="px-2 py-1 rounded border border-slate-200 text-slate-600 hover:bg-red-50 hover:border-red-300 hover:text-red-700 disabled:opacity-50"
              onClick={() => correctionMutation.mutate(NOT_A_BIRD)}
              disabled={correctionMutation.isPending}
              title="Mark as not a bird (false positive)"
              aria-label="Mark as not a bird"
            >
              🚫
            </button>
          </div>
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
