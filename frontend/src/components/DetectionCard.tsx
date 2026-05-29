import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { submitCorrection, type Detection } from "../lib/api";
import AudioBadge from "./AudioBadge";
import SpeciesPicker, { NOT_A_BIRD } from "./SpeciesPicker";

type DetectionCardProps = {
  detection: Detection;
  compact?: boolean;
  // When `selected` is non-undefined the card opts into bulk-select mode:
  // the checkbox appears, the card gains a ring when selected, and tapping
  // the image toggles selection. When undefined, the card is "view-only"
  // (which is how we render in any context without a bulk-action bar).
  selected?: boolean;
  onToggleSelect?: () => void;
};

export default function DetectionCard({
  detection,
  compact = false,
  selected,
  onToggleSelect,
}: DetectionCardProps) {
  const selectable = selected !== undefined && onToggleSelect !== undefined;
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

  const ringClass = selected ? "ring-2 ring-forest" : "ring-1 ring-transparent";

  return (
    <div className={`bg-white rounded-lg shadow-sm overflow-hidden flex flex-col relative ${ringClass}`}>
      {selectable && (
        // Position the checkbox over the top-left of the image. Larger
        // hit-target than a default 16-px input so mobile thumbs hit it
        // reliably without zooming. Backdrop ensures contrast on bright
        // crops.
        <label
          className="absolute top-1 left-1 z-10 flex items-center justify-center w-7 h-7 rounded bg-white/80 backdrop-blur-sm cursor-pointer"
          onClick={(e) => e.stopPropagation()}
          aria-label={selected ? "Deselect" : "Select"}
        >
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="w-5 h-5 accent-forest cursor-pointer"
          />
        </label>
      )}
      <img
        src={detection.crop_url}
        alt={detection.species ?? "bird"}
        className={`w-full object-cover aspect-[4/3] ${selectable ? "cursor-pointer" : ""}`}
        loading="lazy"
        onClick={selectable ? onToggleSelect : undefined}
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
        // The classifier's full top-K is already on the wire in raw_predictions.
        // Surfacing it pinned at the top of the picker means the correct ID is
        // often a single tap away when the top-1 is wrong (#2-#5 frequently
        // contain it for borderline crops).
        suggestions={detection.raw_predictions}
        onSelect={(name) => correctionMutation.mutate(name)}
        onCancel={() => setPickerOpen(false)}
      />
    </div>
  );
}
