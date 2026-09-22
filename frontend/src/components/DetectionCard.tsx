import { useState, useRef, useEffect, useLayoutEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import {
  confirmClassifierLabel,
  submitCorrection,
  undoCorrection,
  type Detection,
} from "../lib/api";
import AudioBadge from "./AudioBadge";
import ImageZoom from "./ImageZoom";
import SpeciesPicker, { NOT_A_BIRD, POOR_QUALITY } from "./SpeciesPicker";
import { useToast } from "./Toast";
import { BanIcon, CheckIcon, CloseIcon, EditIcon, FogIcon, InfoIcon, ZoomIcon } from "./FieldIcons";

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
  // When >1, this card stands in for a collapsed burst of crops from the
  // same visit (the "Best only" view). Drives a small "1 of N" badge.
  seriesCount?: number;
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
  seriesCount = 1,
}: DetectionCardProps) {
  const selectable = selected !== undefined && onToggleSelect !== undefined;
  const showToast = useToast();
  // Show CAPTURE time (when the camera saw the bird). The API tags
  // captured_at as naive UTC; append 'Z' so JS parses it as UTC and
  // toLocaleString converts to the viewer's zone.
  const time = new Date(detection.captured_at + "Z").toLocaleString();
  const [pickerOpen, setPickerOpen] = useState(false);
  const [zoomOpen, setZoomOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [detailsHovered, setDetailsHovered] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number; width: number } | null>(null);

  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const queryClient = useQueryClient();

  const isDetailsVisible = detailsOpen || detailsHovered;

  const startCloseTimer = () => {
    if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
    closeTimerRef.current = setTimeout(() => {
      setDetailsHovered(false);
      setDetailsOpen((open) => {
        if (!open) setCoords(null);
        return open;
      });
    }, 150);
  };

  const clearCloseTimer = () => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  };

  const computeCoords = () => {
    if (!triggerRef.current) return null;
    const triggerRect = triggerRef.current.getBoundingClientRect();
    const tooltipWidth = Math.min(264, window.innerWidth - 24);
    let top = triggerRect.bottom + 6;
    if (top + 240 > window.innerHeight - 8 && triggerRect.top - 240 - 6 > 8) {
      top = triggerRect.top - 240 - 6;
    }
    let left = triggerRect.right - tooltipWidth;
    if (left < 12) left = 12;
    if (left + tooltipWidth > window.innerWidth - 12) {
      left = window.innerWidth - 12 - tooltipWidth;
    }
    return { top, left, width: tooltipWidth };
  };

  const showDetails = (pinned = false) => {
    clearCloseTimer();
    const calculated = computeCoords();
    if (calculated) {
      setCoords(calculated);
    }
    if (pinned) {
      setDetailsOpen(true);
    } else {
      setDetailsHovered(true);
    }
  };

  const closeDetails = useCallback(() => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
    setDetailsOpen(false);
    setDetailsHovered(false);
    setCoords(null);
  }, []);

  useLayoutEffect(() => {
    if (!isDetailsVisible || !coords || !triggerRef.current || !tooltipRef.current) return;
    const tooltipHeight = tooltipRef.current.offsetHeight;
    const triggerRect = triggerRef.current.getBoundingClientRect();
    if (coords.top + tooltipHeight > window.innerHeight - 8 && triggerRect.top - tooltipHeight - 6 > 8) {
      const flippedTop = triggerRect.top - tooltipHeight - 6;
      if (Math.abs(flippedTop - coords.top) > 2) {
        setCoords((prev) => (prev ? { ...prev, top: flippedTop } : null));
      }
    }
  }, [isDetailsVisible, coords]);

  useEffect(() => {
    if (!isDetailsVisible) return;

    const onScrollOrResize = () => {
      closeDetails();
    };

    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Node;
      if (
        triggerRef.current &&
        !triggerRef.current.contains(target) &&
        tooltipRef.current &&
        !tooltipRef.current.contains(target)
      ) {
        closeDetails();
      }
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        closeDetails();
      }
    };

    window.addEventListener("scroll", onScrollOrResize, { passive: true });
    window.addEventListener("resize", onScrollOrResize, { passive: true });
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);

    return () => {
      window.removeEventListener("scroll", onScrollOrResize);
      window.removeEventListener("resize", onScrollOrResize);
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isDetailsVisible, closeDetails]);

  useEffect(() => {
    return () => {
      if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
    };
  }, []);

  const identified = detection.species !== null;
  const pct = Math.round(detection.confidence * 100);
  const tier = confTier(detection.confidence);

  // Undo backs the toast shown after any correction/confirmation: delete
  // the Correction row that action created and restore the prior species.
  // The action card usually unmounts on the post-action refetch (it no
  // longer matches the feed/review filter), so undo can't run through a
  // card-scoped mutation — it calls the API directly and leans on the
  // context-level queryClient + toast, which both outlive the card.
  const offerUndo = (
    message: string,
    correction_id: number,
    restore_species_id: number | null,
  ) => {
    showToast(message, {
      label: "Undo",
      onAction: () => {
        undoCorrection(correction_id, restore_species_id)
          .then(() => {
            queryClient.invalidateQueries({ queryKey: ["detections"] });
            showToast("Undone");
          })
          .catch(() => showToast("Couldn't undo — try refreshing"));
      },
    });
  };

  const correctionMutation = useMutation({
    mutationFn: (species: string) => submitCorrection(detection.id, species),
    onSuccess: (data, species) => {
      queryClient.invalidateQueries({ queryKey: ["detections"] });
      setPickerOpen(false);
      const verb =
        species === NOT_A_BIRD
          ? "Marked “Not a bird”"
          : species === POOR_QUALITY
            ? "Marked poor quality"
            : `Changed to ${data.species}`;
      offerUndo(verb, data.correction_id, data.prev_species_id);
    },
  });

  const isAwaitingClassifierReview =
    reviewMode && detection.species !== null && detection.correction_source === null;
  const confirmMutation = useMutation({
    mutationFn: () => confirmClassifierLabel(detection.id),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["detections"] });
      offerUndo(
        data.species ? `Confirmed ${data.species}` : "Confirmed",
        data.correction_id,
        data.prev_species_id,
      );
    },
  });

  // Outline state: selection (leaf), production-classifier review (soft
  // leaf), or none.
  const ringClass = selected
    ? "ring-2 ring-leaf"
    : isAwaitingClassifierReview
      ? "ring-1 ring-leaf/40"
      : "ring-1 ring-transparent";

  return (
    <div
      className={`fg-card fg-liftable flex flex-col relative ${ringClass}`}
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
        className={`group relative w-full aspect-[4/3] overflow-hidden rounded-t-[calc(var(--radius)-1px)] bg-panel ${
          selectable ? "cursor-pointer" : "cursor-zoom-in"
        }`}
        title={selectable ? undefined : "Click to inspect & compare processing variants (Raw, Chroma, CLAHE, Sharpen)"}
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
        {seriesCount > 1 && (
          <span
            className="absolute left-2 bottom-2 z-[3] rounded-full bg-surface/85 px-2 py-0.5 text-[10px] font-semibold text-muted backdrop-blur-sm"
            title={`Best of ${seriesCount} crops from this visit`}
          >
            1 of {seriesCount}
          </span>
        )}
        <div className={`fg-confbar tier-${tier}`} aria-hidden>
          <span style={{ width: `${Math.max(6, pct)}%` }} />
        </div>
      </div>

      {/* Review mode: stack the proposed species' reference photo below the
          crop at the same size for a direct A/B compare. */}
      {isAwaitingClassifierReview &&
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

      <div className={`p-3 flex flex-col flex-1 ${compact ? "text-xs" : "text-sm"}`}>
        <div className="flex items-start justify-between gap-1 leading-tight">
          <div className="min-w-0 flex-1">
            {/* Identified species link to their plate page (all sightings of
                that species); sentinels and Unidentified stay plain text. */}
            {identified && detection.species_id != null ? (
              <Link
                to={`/species/${detection.species_id}`}
                className={`font-serif text-ink font-medium hover:text-leaf hover:underline underline-offset-2 transition-colors inline ${
                  compact ? "text-sm" : "text-[16px]"
                }`}
              >
                {detection.species}
              </Link>
            ) : (
              <span
                className={`font-serif inline ${
                  identified ? "text-ink font-medium" : "text-muted italic"
                } ${compact ? "text-sm" : "text-[16px]"}`}
              >
                {detection.species ?? "Unidentified"}
              </span>
            )}
          </div>
          <button
            type="button"
            ref={triggerRef}
            onClick={(e) => {
              e.stopPropagation();
              if (detailsOpen) {
                closeDetails();
              } else {
                showDetails(true);
              }
            }}
            onMouseEnter={() => {
              showDetails(false);
            }}
            onMouseLeave={startCloseTimer}
            onFocus={() => {
              showDetails(false);
            }}
            onBlur={startCloseTimer}
            aria-label="Detection details"
            aria-expanded={isDetailsVisible}
            title="View details & metadata"
            className={`p-1 -mr-1 -mt-0.5 rounded transition-colors shrink-0 ${
              isDetailsVisible
                ? "text-leaf bg-panel"
                : "text-muted hover:text-ink focus-visible:text-leaf"
            }`}
          >
            <InfoIcon size={15} />
          </button>
        </div>

        <div className="mt-1 flex items-center gap-1.5 text-xs text-muted">
          {pct > 0 ? (
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
          ) : detection.correction_source?.startsWith("llm-claude") ? (
            <span className="fg-overline text-leaf" title="Labeled by Claude, not the on-device classifier">
              AI label
            </span>
          ) : null}
          <span className="text-faint tnum">{time}</span>
        </div>

        {/* Quick-review row: Confirm / NAB / Poor quality / Change.
            Four columns. Poor quality retires the crop from any review
            queue forever (same Correction-to-sentinel mechanism as NAB);
            semantically "this image will never be identifiable, stop
            showing it to me." Uses the FogIcon (dashed rules) and a
            sand-tinted hover to distinguish from NAB's rust. */}
        {!compact && isAwaitingClassifierReview && (
          <div className="mt-auto pt-2.5 grid grid-cols-4 gap-1.5 text-xs">
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
        {!compact && !isAwaitingClassifierReview && (
          <div className="mt-auto pt-2.5 flex items-center justify-between gap-2 text-xs">
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

      {isDetailsVisible &&
        coords &&
        createPortal(
          <div
            ref={tooltipRef}
            role="tooltip"
            onMouseEnter={clearCloseTimer}
            onMouseLeave={startCloseTimer}
            style={{
              position: "fixed",
              top: `${coords.top}px`,
              left: `${coords.left}px`,
              width: `${coords.width}px`,
            }}
            className="z-50 p-3 rounded-card bg-surface/95 backdrop-blur-md border border-line shadow-pop text-ink flex flex-col gap-2.5 select-none"
          >
            {/* Header with Title and close button */}
            <div className="flex items-center justify-between border-b border-line/60 pb-1.5">
              <span className="fg-overline text-muted">Detection Details</span>
              <button
                type="button"
                onClick={closeDetails}
                className="text-muted hover:text-ink p-0.5 -mr-1 rounded"
                aria-label="Close details"
              >
                <CloseIcon size={12} />
              </button>
            </div>

            {/* Audio Confirmation */}
            {detection.audio_confirmed && (
              <div className="flex items-center gap-2 text-leaf font-medium text-xs">
                <AudioBadge />
                <span className="text-[11px] text-muted">Heard by Haikubox within 90s</span>
              </div>
            )}

            {/* Enhancements / Computational Photography */}
            {(detection.has_lucky || detection.has_sr || detection.has_sisr) && (
              <div className="space-y-1">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-muted">
                  Enhancement
                </div>
                <div className="flex flex-col gap-1 text-[11px]">
                  {detection.has_lucky && (
                    <div className="flex items-center gap-1.5">
                      <span className="inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-amber-800 dark:text-amber-300 border border-amber-600/25 dark:border-amber-500/35 bg-amber-500/10 leading-none shrink-0">
                        ★ lucky
                      </span>
                      <span className="text-muted">Sharpest burst frame selected</span>
                    </div>
                  )}
                  {detection.has_sr && (
                    <div className="flex items-center gap-1.5">
                      <span className="inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-purple-800 dark:text-purple-300 border border-purple-600/25 dark:border-purple-500/35 bg-purple-500/10 leading-none shrink-0">
                        2× sr
                      </span>
                      <span className="text-muted">Multi-frame super-resolution</span>
                    </div>
                  )}
                  {detection.has_sisr && (
                    <div className="flex items-center gap-1.5">
                      <span className="inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-indigo-800 dark:text-indigo-300 border border-indigo-600/25 dark:border-indigo-500/35 bg-indigo-500/10 leading-none shrink-0">
                        2× neural
                      </span>
                      <span className="text-muted">FSRCNN neural reconstruction</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Quality Metrics */}
            {detection.crop_area_px != null &&
              detection.brightness != null &&
              detection.sharpness != null &&
              (() => {
                const maxDim = Math.round(Math.sqrt(detection.crop_area_px!));
                const isSmall = maxDim < 80;
                const isDark = detection.brightness! < 30;
                const isBlurry = detection.sharpness! < 30;
                const anyBad = isSmall || isDark || isBlurry;
                const cls = (bad: boolean) => (bad ? "text-rust font-semibold" : "text-ink");
                return (
                  <div>
                    <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-muted mb-1">
                      <span>Crop Quality</span>
                      {anyBad && (
                        <span className="text-rust normal-case font-normal text-[10px]">
                          Below threshold
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-x-2 text-[11px] tnum text-muted">
                      <span className={cls(isSmall)}>{maxDim}px</span>
                      <span className="text-line">·</span>
                      <span className={cls(isDark)}>lum {Math.round(detection.brightness!)}</span>
                      <span className="text-line">·</span>
                      <span className={cls(isBlurry)}>shp {Math.round(detection.sharpness!)}</span>
                    </div>
                  </div>
                );
              })()}

            {/* Alternative Predictions */}
            {(() => {
              const alts = (detection.raw_predictions ?? [])
                .filter((p) => p.species && p.species !== detection.species && p.p >= 0.02)
                .slice(0, 3);
              if (alts.length === 0) return null;
              return (
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-muted mb-1">
                    Alternative Predictions
                  </div>
                  <div className="space-y-1">
                    {alts.map((a) => (
                      <div key={a.species} className="flex justify-between items-center text-[11px]">
                        <span className="truncate text-ink mr-2">{a.species}</span>
                        <span className="text-muted tnum shrink-0 font-medium">
                          {Math.round(a.p * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* AI Rationale / Provenance */}
            {detection.correction_source === "llm-claude" && detection.correction_rationale && (
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-wider text-muted mb-0.5">
                  <span className="text-leaf">AI Rationale</span>
                </div>
                <p className="text-[11px] text-ink italic leading-snug">
                  {detection.correction_rationale}
                </p>
              </div>
            )}

            {/* NAB override filter score */}
            {detection.nab_override_p != null && (
              <div className="flex items-center justify-between text-[11px] text-muted">
                <span>NAB filter score:</span>
                <span className="text-rust font-semibold tnum">
                  {Math.round(detection.nab_override_p * 100)}%
                </span>
              </div>
            )}

            {/* Footer with visit and track IDs */}
            <div className="text-[10px] text-faint flex justify-between items-center pt-1.5 border-t border-line/60 tnum">
              <span>Visit #{detection.visit_id}</span>
              <span>Track #{detection.track_id}</span>
            </div>
          </div>,
          document.body
        )}

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
          detectionId={detection.id}
          species={detection.species}
          initialSharpness={detection.sharpness}
          cropAreaPx={detection.crop_area_px}
          brightness={detection.brightness}
          hasLucky={detection.has_lucky}
          hasSr={detection.has_sr}
          hasSisr={detection.has_sisr}
          onClose={() => setZoomOpen(false)}
        />
      )}
    </div>
  );
}
