import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { bulkCorrection } from "../lib/api";
import SpeciesPicker, { NOT_A_BIRD } from "./SpeciesPicker";

type Props = {
  selectedIds: number[];
  onClear: () => void;
};

/**
 * Floating action bar that appears whenever the user has ≥ 1 detection
 * selected via the per-card checkboxes. Provides:
 *   - 🚫 one-tap bulk-mark-as-NAB (the dominant labeling workflow)
 *   - "Label as…" → opens the SpeciesPicker, applies the chosen species
 *     to every selected detection
 *   - Cancel
 *
 * Sits above the bottom nav (which is sticky at `bottom-0`); we anchor
 * at `bottom-14` to clear it.
 */
export default function BulkActionBar({ selectedIds, onClear }: Props) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const queryClient = useQueryClient();

  const bulk = useMutation({
    mutationFn: (species: string) => bulkCorrection(selectedIds, species),
    onSuccess: () => {
      // Optimistic refresh — all the updated detections move state
      // (out of the current view if they were in /feed and got marked NAB;
      // out of /labels if they got re-corrected to a real species).
      queryClient.invalidateQueries({ queryKey: ["detections"] });
      setPickerOpen(false);
      onClear();
    },
  });

  if (selectedIds.length === 0) return null;

  return (
    <>
      <div className="fixed bottom-14 left-0 right-0 z-40 bg-forest text-cream shadow-lg px-3 py-2 flex items-center justify-between gap-2">
        <span className="text-sm font-semibold">
          {selectedIds.length} selected
        </span>
        <div className="flex items-center gap-2">
          <button
            className="px-3 py-1 rounded bg-cream/10 hover:bg-cream/20 text-sm disabled:opacity-50"
            onClick={() => bulk.mutate(NOT_A_BIRD)}
            disabled={bulk.isPending}
            title="Mark all selected as 'Not a bird'"
          >
            {bulk.isPending && bulk.variables === NOT_A_BIRD ? "Saving…" : "🚫 Not a bird"}
          </button>
          <button
            className="px-3 py-1 rounded bg-cream text-forest font-semibold text-sm disabled:opacity-50"
            onClick={() => setPickerOpen(true)}
            disabled={bulk.isPending}
          >
            Label as…
          </button>
          <button
            className="px-2 py-1 text-sm opacity-80 hover:opacity-100"
            onClick={onClear}
            disabled={bulk.isPending}
          >
            Cancel
          </button>
        </div>
      </div>
      {bulk.isError && (
        <div className="fixed bottom-28 left-0 right-0 z-40 px-3 text-center">
          <span className="inline-block bg-red-100 border border-red-300 text-red-800 text-xs px-2 py-1 rounded">
            Bulk correction failed: {(bulk.error as Error).message}
          </span>
        </div>
      )}
      <SpeciesPicker
        open={pickerOpen}
        onSelect={(name) => bulk.mutate(name)}
        onCancel={() => setPickerOpen(false)}
      />
    </>
  );
}
