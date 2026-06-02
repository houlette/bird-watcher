import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import {
  confirmClassifierLabel,
  confirmLlmCorrection,
  submitCorrection,
  type Detection,
} from "../lib/api";
import AudioBadge from "./AudioBadge";
import ImageZoom from "./ImageZoom";
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
  // When true, classifier-labeled-but-unreviewed cards swap the default
  // "Wrong species?" row for a ✓ Confirm / 🚫 NAB / ✏️ Change row, so the
  // user can record true positives. Set by the Feed's "Awaiting review"
  // filter. Other filters (default, species, NAB, etc.) leave the card
  // visually unchanged.
  reviewMode?: boolean;
};

export default function DetectionCard({
  detection,
  compact = false,
  selected,
  onToggleSelect,
  reviewMode = false,
}: DetectionCardProps) {
  const selectable = selected !== undefined && onToggleSelect !== undefined;
  // Show CAPTURE time (when the camera saw the bird), not the row's
  // processing time. The API tags captured_at as naive UTC; JS's Date
  // parser interprets a naive ISO string as local time, so append 'Z'
  // before parsing to treat it as UTC and let toLocaleString convert
  // to the viewer's local zone.
  const time = new Date(detection.captured_at + "Z").toLocaleString();
  const [pickerOpen, setPickerOpen] = useState(false);
  const [zoomOpen, setZoomOpen] = useState(false);
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

  // The ✓ Confirm button has two call-sites with two endpoints:
  //   - MEDIUM LLM review: promotes the existing llm-claude-medium
  //     Correction's source to llm-claude-confirmed.
  //   - Awaiting-review (classifier output, no Correction yet): creates
  //     a fresh Correction with source="user-confirmed" so we can
  //     measure classifier true-positive rate.
  // The card decides which to call based on whether there's already an
  // llm-claude-medium Correction (the former).
  const isLlmMediumReview = detection.correction_source === "llm-claude-medium";
  const isAwaitingClassifierReview =
    reviewMode && detection.species !== null && detection.correction_source === null;
  const confirmMutation = useMutation({
    mutationFn: () =>
      isLlmMediumReview
        ? confirmLlmCorrection(detection.id)
        : confirmClassifierLabel(detection.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["detections"] });
    },
  });

  const ringClass = selected
    ? "ring-2 ring-forest"
    : isLlmMediumReview
      ? "ring-1 ring-amber-300"   // visual cue: distinguish MEDIUM-review cohort from HIGH
      : isAwaitingClassifierReview
        ? "ring-1 ring-blue-300"  // visual cue: production-classifier review
        : "ring-1 ring-transparent";

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
      {/* Crops vary in aspect ratio (a vertically-perched cardinal vs a
          horizontally-strutting pigeon), but the feed grid is uniform 4:3.
          `object-cover` previously clipped tall birds at the head/tail;
          we now `object-contain` the actual crop and fill the surrounding
          letterbox with a heavily-blurred copy of the same image. The
          browser reuses the cached crop_url, so this adds essentially no
          network cost — and visually we get the full bird with a soft,
          ambient backdrop. */}
      <div
        className={`relative w-full aspect-[4/3] overflow-hidden ${selectable ? "cursor-pointer" : "cursor-zoom-in"}`}
        onClick={selectable ? onToggleSelect : () => setZoomOpen(true)}
      >
        <img
          src={detection.crop_url}
          aria-hidden
          // scale-110 hides the blur's edge halo at the card boundary.
          className="absolute inset-0 w-full h-full object-cover blur-2xl scale-110"
          loading="lazy"
        />
        <img
          src={detection.crop_url}
          alt={detection.species ?? "bird"}
          className="relative w-full h-full object-contain"
          loading="lazy"
        />
      </div>
      {/* In review mode (LLM-MEDIUM or Awaiting-review), stack a
          reference photo of the proposed species directly below the
          user's crop at the same width and aspect ratio. Lets the user
          compare two same-size images side-by-side without leaving the
          feed. A small "REF" badge in the corner distinguishes it from
          the user's crop. Falls back to nothing if the species has no
          reference URL (sentinel, family, or unfetched). */}
      {(isLlmMediumReview || isAwaitingClassifierReview) &&
        detection.reference_image_url && (
          <div className="relative w-full aspect-[4/3] overflow-hidden border-t border-slate-200">
            <img
              src={detection.reference_image_url}
              aria-hidden
              className="absolute inset-0 w-full h-full object-cover blur-2xl scale-110 opacity-70"
              loading="lazy"
            />
            <img
              src={detection.reference_image_url}
              alt={detection.species ? `${detection.species} reference photo` : "reference"}
              className="relative w-full h-full object-contain"
              loading="lazy"
              onError={(e) => {
                (e.currentTarget.parentElement as HTMLElement).style.display = "none";
              }}
            />
            <span className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-white/85 text-[10px] font-medium text-slate-600 uppercase tracking-wide">
              ref
            </span>
          </div>
        )}
      <div className={`p-2 ${compact ? "text-xs" : "text-sm"}`}>
        <div className="flex items-center justify-between gap-2">
          <span className="font-semibold">{detection.species ?? "Unidentified"}</span>
          {detection.audio_confirmed && <AudioBadge />}
        </div>
        <div className="text-slate-500">{Math.round(detection.confidence * 100)}% · {time}</div>
        {/* When the label was generated by the LLM backlog-classifier pass,
            surface the model's one-line rationale so the user can spot-check
            without having to open and stare at every crop. Hidden when the
            label came from the user or the production classifier. */}
        {detection.correction_source === "llm-claude" && detection.correction_rationale && (
          <div
            className="text-xs text-slate-500 mt-0.5 italic"
            title="LLM rationale — open SpeciesPicker if it's wrong"
          >
            <span aria-hidden>✨ </span>
            <span>{detection.correction_rationale}</span>
          </div>
        )}
        {/* Classifier's next-best guesses. dennisjooo's top-5 hit rate is much
            higher than top-1, so showing the next 2 candidates with their
            probabilities tells the user when the model was borderline and
            often surfaces the right answer at a glance. Filter to ≥ 2% to
            skip the noise tail; cap at 2 to keep the card compact. */}
        {(() => {
          const alts = (detection.raw_predictions ?? [])
            .filter((p) => p.species && p.species !== detection.species && p.p >= 0.02)
            .slice(0, 2);
          if (alts.length === 0) return null;
          // Two-line clamp instead of single-line truncate: the
          // alternates are useful for sanity-checking the top-1, so
          // the user would rather see "Northern Goshawk 14% · Cooper's
          // Hawk 9%" than "Northern Goshawk…".
          return (
            <div className="text-xs text-slate-400 mt-0.5 line-clamp-2">
              {alts.map((a) => `${a.species} ${Math.round(a.p * 100)}%`).join(" · ")}
            </div>
          );
        })()}

        {/* Quick-review action row used by two filters: LLM-labeled MEDIUM
            and "Awaiting review" (production classifier output). Same buttons,
            same semantics; the ✓ mutation routes to the appropriate endpoint
            based on the card's state. Three equal-width grid columns so the
            pencil never gets clipped on narrow cards (the lg:grid-cols-5 feed
            gives each card ~180px). Confirm becomes icon-only — green border
            + checkmark + tooltip carry the meaning at a fraction of the
            horizontal cost. */}
        {!compact && (isLlmMediumReview || isAwaitingClassifierReview) && (
          <div className="mt-2 grid grid-cols-3 gap-1 text-xs">
            <button
              className="px-1 py-1 rounded border border-emerald-300 text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
              onClick={() => confirmMutation.mutate()}
              disabled={confirmMutation.isPending || correctionMutation.isPending}
              title="Confirm — keep this label"
              aria-label="Confirm"
            >
              {confirmMutation.isPending ? "…" : "✓"}
            </button>
            <button
              className="px-1 py-1 rounded border border-slate-200 text-slate-600 hover:bg-red-50 hover:border-red-300 hover:text-red-700 disabled:opacity-50"
              onClick={() => correctionMutation.mutate(NOT_A_BIRD)}
              disabled={confirmMutation.isPending || correctionMutation.isPending}
              title="Mark as not a bird (false positive)"
              aria-label="Mark as not a bird"
            >
              🚫
            </button>
            <button
              className="px-1 py-1 rounded border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
              onClick={() => setPickerOpen(true)}
              disabled={confirmMutation.isPending || correctionMutation.isPending}
              title="Change species"
              aria-label="Change species"
            >
              ✏️
            </button>
          </div>
        )}
        {!compact && !isLlmMediumReview && !isAwaitingClassifierReview && (
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
        {confirmMutation.isError && (
          <p className="text-xs text-red-600 mt-1">
            Couldn't confirm: {(confirmMutation.error as Error).message}
          </p>
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
        // Enables the side-by-side compare strip: user's crop pinned next
        // to the reference photo of whichever species they're hovering.
        cropUrl={detection.crop_url}
        onSelect={(name) => correctionMutation.mutate(name)}
        onCancel={() => setPickerOpen(false)}
      />
      {zoomOpen && (
        <ImageZoom
          src={detection.crop_url}
          alt={detection.species ?? "bird"}
          onClose={() => setZoomOpen(false)}
        />
      )}
    </div>
  );
}
