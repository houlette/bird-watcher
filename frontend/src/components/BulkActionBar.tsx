import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { bulkCorrection } from "../lib/api";
import SpeciesPicker, { NOT_A_BIRD } from "./SpeciesPicker";
import { BanIcon } from "./FieldIcons";

type Props = {
  selectedIds: number[];
  onClear: () => void;
};

/**
 * Floating action bar that appears whenever the user has ≥ 1 detection
 * selected via the per-card checkboxes. Provides:
 *   - one-tap bulk-mark-as-NAB (the dominant labeling workflow)
 *   - "Label as…" → opens the SpeciesPicker, applies the chosen species
 *     to every selected detection
 *   - Cancel
 *
 * Styled as the field-guide "command pill": a floating ink slab centred
 * above the sticky bottom nav, echoing the toast and zoom-modal language.
 */
export default function BulkActionBar({ selectedIds, onClear }: Props) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const queryClient = useQueryClient();

  const bulk = useMutation({
    mutationFn: (species: string) => bulkCorrection(selectedIds, species),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["detections"] });
      setPickerOpen(false);
      onClear();
    },
  });

  if (selectedIds.length === 0) return null;

  return (
    <>
      <div className="fixed left-1/2 -translate-x-1/2 bottom-[5.5rem] z-40 w-[min(520px,calc(100%-28px))] flex items-center gap-3 rounded-2xl px-4 py-3 bg-ink text-paper shadow-[0_18px_40px_-18px_rgba(0,0,0,.5)]">
        <span className="text-sm">
          <strong className="font-bold tnum">{selectedIds.length}</strong> selected
        </span>
        <div className="flex-1" />
        <button
          className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold border border-paper/25 text-paper hover:text-rust hover:border-rust transition-colors disabled:opacity-50"
          onClick={() => bulk.mutate(NOT_A_BIRD)}
          disabled={bulk.isPending}
          title="Mark all selected as 'Not a bird'"
        >
          <BanIcon size={15} />
          {bulk.isPending && bulk.variables === NOT_A_BIRD ? "Saving…" : "Not a bird"}
        </button>
        <button
          className="rounded-lg px-3.5 py-1.5 text-sm font-semibold text-surface disabled:opacity-50"
          style={{ background: "var(--accent)" }}
          onClick={() => setPickerOpen(true)}
          disabled={bulk.isPending}
        >
          {bulk.isPending && bulk.variables !== NOT_A_BIRD ? "Saving…" : "Label as…"}
        </button>
        {/* Cancel is intentionally NEVER disabled — it's the escape hatch.
            A slow or failed bulk request used to leave every button
            (including this one) disabled, stranding the user with no way
            out. reset() clears a stuck/errored mutation so re-entering
            batch mode starts clean. */}
        <button
          className="px-1.5 py-1 text-sm text-paper/70 hover:text-paper transition-colors"
          onClick={() => {
            bulk.reset();
            onClear();
          }}
        >
          Cancel
        </button>
      </div>
      {bulk.isError && (
        <div className="fixed left-1/2 -translate-x-1/2 bottom-[9rem] z-40 px-3 text-center">
          <span className="inline-block rounded-lg border border-rust/40 bg-[color-mix(in_oklab,var(--rust)_14%,var(--card))] text-rust text-xs px-2.5 py-1">
            Bulk correction failed: {(bulk.error as Error).message}
          </span>
        </div>
      )}
      <SpeciesPicker
        open={pickerOpen}
        onSelect={(name) => {
          // Close the picker immediately so the user gets feedback that
          // their pick registered, and so the bar's pending state / any
          // error toast (both at z-40) aren't hidden behind the picker's
          // z-50 backdrop. The mutation's onSuccess also closes it, but
          // doing it here covers the slow / error paths too.
          setPickerOpen(false);
          bulk.mutate(name);
        }}
        onCancel={() => setPickerOpen(false)}
      />
    </>
  );
}
