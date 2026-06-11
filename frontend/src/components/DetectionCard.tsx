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
import SpeciesPicker, { NOT_A_BIRD, POOR_QUALITY } from "./SpeciesPicker";
import { BanIcon, CheckIcon, EditIcon, FogIcon, ZoomIcon } from "./FieldIcons";

type DetectionCardProps = {
  detection: Detection;
  compact?: boolean;
  // When `selected` is non-undefined the card opts into bulk-select mode:
  // the checkbox appears, the card gains a ring when selected, and tapping
  // the image toggles selection. When undefined, the card is "view-only".
  selected?: boolean;
  onToggleSelect?: () => void;
  // When true, classifier-labeled-but-unreviewed cards swap the default
  // "Wrong species?" row for a Confirm / NAB / Change row.
  reviewMode?: boolean;
};

// Confidence → tier. Drives the ribbon under the crop and the dot in the
// meta line: leaf (high) / muted-leaf (mid) / rust (low).
function confTier(p: number): "high" | "mid" | "low" {
  if (p >= 0.85) return "high";
  if (p >= 0.6) return "mid";
  return "low";
}

export default function DetectionCard({
  detection,
  compact = false,
  selected,
  onToggleSelect,
  reviewMode = false,
}: DetectionCardProps) {
  const selectable = selected !== undefined && onToggleSelect !== undefined;
  // Show CAPTURE time (when the camera saw the bird). The API tags
  // captured_at as naive UTC; append 'Z' so JS parses it as UTC and
  // toLocaleString converts to the viewer's zone.
  const time = new Date(detection.captured_at + "Z").toLocaleString();
  const [pickerOpen, setPickerOpen] = useState(false);
  const [zoomOpen, setZoomOpen] = useState(false);
  const queryClient = useQueryClient();

  const identified = detection.species !== null;
  const pct = Math.round(detection.confidence * 100);
  const tier = confTier(detection.confidence);

  const correctionMutation = useMutation({
    mutationFn: (species: string) => submitCorrection(detection.id, species),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["detections"] });
      setPickerOpen(false);
    },
  });

  // The Confirm button routes to one of two endpoints (see original notes).
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

  // Outline state: selection (leaf), MEDIUM-review cohort (rust/warning),
  // production-classifier review (soft leaf), or none.
  const ringClass = selected
    ? "ring-2 ring-leaf"
    : isLlmMediumReview
      ? "ring-1 ring-rust/50"
      : isAwaitingClassifierReview
        ? "ring-1 ring-leaf/40"
        : "ring-1 ring-transparent";

  return (
    <div
      className={`fg-card fg-liftable overflow-hidden flex flex-col relative ${ringClass}`}
    >
      {selectable && (
        <label
          className="absolute top-2 left-2 z-10 flex items-center justify-center w-6 h-6 rounded-md bg-surface/85 backdrop-blur-sm border border-line cursor-pointer"
          onClick={(e) => e.stopPropagation()}
          aria-label={selected ? "Deselect" : "Select"}
        >
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="w-4 h-4 fg-range cursor-pointer"
          />
        </label>
      )}

      {/* Crop: heavily-blurred copy fills the 4:3 letterbox; the real crop is
          object-contain on top so tall/wide birds aren't clipped. A leaf/rust
          confidence ribbon runs along the bottom edge. */}
      <div
        className={`group relative w-full aspect-[4/3] overflow-hidden bg-panel ${
          selectable ? "cursor-pointer" : "cursor-zoom-in"
        }`}
        onClick={selectable ? onToggleSelect : () => setZoomOpen(true)}
      >
        <img
          src={detection.crop_url}
          aria-hidden
          className="absolute inset-0 w-full h-full object-cover blur-2xl scale-110 opacity-30 saturate-[.85]"
          loading="lazy"
        />
        <img
          src={detection.crop_url}
          alt={detection.species ?? "bird"}
          className="relative w-full h-full object-contain"
          loading="lazy"
        />
        {!selectable && (
          <span className="absolute right-2 bottom-2 z-[3] grid place-items-center w-6 h-6 rounded-full bg-surface/80 text-ink backdrop-blur-sm opacity-0 translate-y-1 transition group-hover:opacity-100 group-hover:translate-y-0">
            <ZoomIcon size={14} />
          </span>
        )}
        <div className={`fg-confbar tier-${tier}`} aria-hidden>
          <span style={{ width: `${Math.max(6, pct)}%` }} />
        </div>
      </div>

      {/* Review mode: stack the proposed species' reference photo below the
          crop at the same size for a direct A/B compare. */}
      {(isLlmMediumReview || isAwaitingClassifierReview) &&
        detection.reference_image_url && (
          <div className="relative w-full aspect-[4/3] overflow-hidden border-t border-line">
            <img
              src={detection.reference_image_url}
              aria-hidden
              className="absolute inset-0 w-full h-full object-cover blur-2xl scale-110 opacity-50"
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
            <span className="fg-overline absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded bg-surface/85">
              ref
            </span>
          </div>
        )}

      <div className={`p-3 ${compact ? "text-xs" : "text-sm"}`}>
        <div className="flex items-start justify-between gap-2">
          <span
            className={`font-serif leading-tight ${
              identified ? "text-ink font-medium" : "text-muted italic"
            } ${compact ? "text-sm" : "text-[16px]"}`}
          >
            {detection.species ?? "Unidentified"}
          </span>
          {detection.audio_confirmed && <AudioBadge />}
        </div>

        <div className="mt-1.5 flex items-center gap-2 text-xs text-muted">
          <span
            className={`inline-flex items-center gap-1 font-semibold tnum ${
              tier === "low" ? "text-rust" : tier === "mid" ? "text-muted" : "text-ink"
            }`}
          >
            <i
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background:
                  tier === "low"
                    ? "var(--rust)"
                    : tier === "mid"
                      ? "color-mix(in oklab, var(--accent) 55%, var(--faint))"
                      : "var(--accent)",
              }}
            />
            {pct}%
          </span>
          <span className="text-faint tnum">{time}</span>
        </div>

        {/* Quality footer — only when all three metrics are populated. Each
            turns rust when it crosses the "bad" threshold (80 / 30 / 30). */}
        {detection.crop_area_px != null &&
          detection.brightness != null &&
          detection.sharpness != null &&
          (() => {
            const maxDim = Math.round(Math.sqrt(detection.crop_area_px!));
            const isSmall = maxDim < 80;
            const isDark = detection.brightness! < 30;
            const isBlurry = detection.sharpness! < 30;
            const anyBad = isSmall || isDark || isBlurry;
            const cls = (bad: boolean) =>
              bad ? "text-rust font-semibold" : "text-faint";
            return (
              <div
                className="mt-1.5 flex items-center gap-2.5 text-[10px] tnum"
                title="Per-crop quality: pixel size · mean brightness 0-255 · Laplacian-variance sharpness. Rust = below the 'bad' threshold (80 / 30 / 30)."
              >
                {anyBad && (
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ background: "var(--rust)" }}
                    aria-hidden
                  />
                )}
                <span className={cls(isSmall)}>{maxDim}px</span>
                <span className="text-line">·</span>
                <span className={cls(isDark)}>lum {Math.round(detection.brightness!)}</span>
                <span className="text-line">·</span>
                <span className={cls(isBlurry)}>shp {Math.round(detection.sharpness!)}</span>
              </div>
            );
          })()}

        {/* LLM rationale for backlog-classifier labels. */}
        {detection.correction_source === "llm-claude" &&
          detection.correction_rationale && (
            <div
              className="mt-1.5 flex gap-1.5 text-xs text-muted italic"
              title="LLM rationale — open the species picker if it's wrong"
            >
              <span className="not-italic fg-overline shrink-0 mt-px text-leaf">AI</span>
              <span>{detection.correction_rationale}</span>
            </div>
          )}

        {/* Classifier's next-best guesses (top-1 excluded). */}
        {(() => {
          const alts = (detection.raw_predictions ?? [])
            .filter((p) => p.species && p.species !== detection.species && p.p >= 0.02)
            .slice(0, 2);
          if (alts.length === 0) return null;
          return (
            <div className="mt-1 text-xs text-faint line-clamp-2 tnum">
              {alts.map((a) => `${a.species} ${Math.round(a.p * 100)}%`).join(" · ")}
            </div>
          );
        })()}

        {/* Quick-review row: Confirm / NAB / Poor quality / Change.
            Four columns. Poor quality retires the crop from any review
            queue forever (same Correction-to-sentinel mechanism as NAB);
            semantically "this image will never be identifiable, stop
            showing it to me." Uses the FogIcon (dashed rules) and a
            sand-tinted hover to distinguish from NAB's rust. */}
        {!compact && (isLlmMediumReview || isAwaitingClassifierReview) && (
          <div className="mt-2.5 grid grid-cols-4 gap-1.5 text-xs">
            <button
              className="inline-flex items-center justify-center gap-1 px-1 py-1.5 rounded-md border text-leaf disabled:opacity-50 transition-colors hover:bg-[color-mix(in_oklab,var(--accent)_12%,transparent)]"
              style={{ borderColor: "color-mix(in oklab, var(--accent) 40%, var(--line))" }}
              onClick={() => confirmMutation.mutate()}
              disabled={confirmMutation.isPending || correctionMutation.isPending}
              title="Confirm — keep this label"
              aria-label="Confirm"
            >
              {confirmMutation.isPending ? "…" : <CheckIcon size={15} />}
            </button>
            <button
              className="inline-flex items-center justify-center px-1 py-1.5 rounded-md border border-line text-muted hover:text-rust hover:border-rust disabled:opacity-50 transition-colors"
              onClick={() => correctionMutation.mutate(NOT_A_BIRD)}
              disabled={confirmMutation.isPending || correctionMutation.isPending}
              title="Mark as not a bird (false positive)"
              aria-label="Mark as not a bird"
            >
              <BanIcon size={15} />
            </button>
            <button
              className="inline-flex items-center justify-center px-1 py-1.5 rounded-md border border-line text-muted disabled:opacity-50 transition-colors hover:text-[color:var(--sand,#bc8a3e)] hover:border-[color:var(--sand,#bc8a3e)]"
              onClick={() => correctionMutation.mutate(POOR_QUALITY)}
              disabled={confirmMutation.isPending || correctionMutation.isPending}
              title="Poor quality — too blurry / small / dark to ever ID. Retires from review queues."
              aria-label="Mark as poor quality"
            >
              <FogIcon size={15} />
            </button>
            <button
              className="inline-flex items-center justify-center px-1 py-1.5 rounded-md border border-line text-muted hover:text-ink disabled:opacity-50 transition-colors"
              onClick={() => setPickerOpen(true)}
              disabled={confirmMutation.isPending || correctionMutation.isPending}
              title="Change species"
              aria-label="Change species"
            >
              <EditIcon size={15} />
            </button>
          </div>
        )}

        {/* Default row: Wrong species? + one-tap NAB. */}
        {!compact && !isLlmMediumReview && !isAwaitingClassifierReview && (
          <div className="mt-2.5 flex items-center justify-between gap-2 text-xs">
            <button
              className="text-muted hover:text-leaf underline underline-offset-2 disabled:opacity-50 transition-colors"
              onClick={() => setPickerOpen(true)}
              disabled={correctionMutation.isPending}
            >
              {correctionMutation.isPending ? "Saving…" : "Wrong species?"}
            </button>
            <button
              className="inline-flex items-center justify-center px-2 py-1 rounded-md border border-line text-muted hover:text-rust hover:border-rust disabled:opacity-50 transition-colors"
              onClick={() => correctionMutation.mutate(NOT_A_BIRD)}
              disabled={correctionMutation.isPending}
              title="Mark as not a bird (false positive)"
              aria-label="Mark as not a bird"
            >
              <BanIcon size={14} />
            </button>
          </div>
        )}

        {confirmMutation.isError && (
          <p className="text-xs text-rust mt-1.5">
            Couldn't confirm: {(confirmMutation.error as Error).message}
          </p>
        )}
        {correctionMutation.isError && (
          <p className="text-xs text-rust mt-1.5">
            Couldn't save correction: {(correctionMutation.error as Error).message}
          </p>
        )}
      </div>

      <SpeciesPicker
        open={pickerOpen}
        current={detection.species}
        suggestions={detection.raw_predictions}
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
