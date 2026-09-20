import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { CloseIcon } from "./FieldIcons";
import { fetchCropVariantsMeta, getCropVariantUrl } from "../lib/api";

type Props = {
  src: string;
  alt: string;
  detectionId?: number;
  species?: string | null;
  initialSharpness?: number | null;
  cropAreaPx?: number | null;
  brightness?: number | null;
  onClose: () => void;
};

type PresetId = "polish" | "raw" | "chroma_only" | "clahe_only" | "sharpen_only" | "custom";
type ZoomMode = "fit" | "1x" | "2x" | "4x";

/**
 * Full-screen computational photography inspection studio for crops.
 *
 * Allows toggling between combinations of techniques:
 *   - Chroma-guided filter (He et al.) in YCrCb: removes 4:2:0 subsampling bleed & color noise
 *   - CLAHE: dynamic range lighting normalization on L channel
 *   - Bilateral unsharp masking: plumage edge sharpening without halos
 *   - Lucky imaging frame compare (when lucky imaging improved the detection)
 *   - Instant A/B hold-to-compare with Raw (via Spacebar or on-screen button)
 *   - 1x/2x/4x magnification with pixelated rendering for barbule-level inspection
 */
export default function ImageZoom({
  src,
  alt,
  detectionId,
  species,
  initialSharpness,
  brightness,
  onClose,
}: Props) {
  // Technique toggle states
  const [chroma, setChroma] = useState(true);
  const [clahe, setClahe] = useState(true);
  const [sharpen, setSharpen] = useState(true);
  const [source, setSource] = useState<"lucky" | "initial">("lucky");
  const [hasInitial, setHasInitial] = useState(false);

  // Comparison & View states
  const [isComparing, setIsComparing] = useState(false);
  const [zoomMode, setZoomMode] = useState<ZoomMode>("fit");
  const [pixelated, setPixelated] = useState(true);

  // Natural image dimensions
  const [naturalDims, setNaturalDims] = useState<{ w: number; h: number } | null>(null);

  // Determine active preset
  let activePreset: PresetId = "custom";
  if (chroma && clahe && sharpen) activePreset = "polish";
  else if (!chroma && !clahe && !sharpen) activePreset = "raw";
  else if (chroma && !clahe && !sharpen) activePreset = "chroma_only";
  else if (!chroma && clahe && !sharpen) activePreset = "clahe_only";
  else if (!chroma && !clahe && sharpen) activePreset = "sharpen_only";

  // Pre-fetch metadata & standard presets into browser cache
  useEffect(() => {
    if (!detectionId) return;
    fetchCropVariantsMeta(detectionId)
      .then((meta) => {
        if (meta.has_initial) setHasInitial(true);
        // Preload standard preset images so toggling is 100% instantaneous
        ["polish", "raw", "chroma_only", "clahe_only", "sharpen_only"].forEach((p) => {
          const img = new Image();
          img.src = getCropVariantUrl(detectionId, { preset: p });
        });
      })
      .catch(() => {});
  }, [detectionId]);

  // Keyboard navigation
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.code === "Space" || e.key === "\\") {
        if (!e.repeat) {
          e.preventDefault();
          setIsComparing(true);
        }
        return;
      }
      if (e.key === "1") {
        setChroma(true);
        setClahe(true);
        setSharpen(true);
      } else if (e.key === "2") {
        setChroma(false);
        setClahe(false);
        setSharpen(false);
      } else if (e.key === "3") {
        setChroma(true);
        setClahe(false);
        setSharpen(false);
      } else if (e.key === "4") {
        setChroma(false);
        setClahe(true);
        setSharpen(false);
      } else if (e.key === "5") {
        setChroma(false);
        setClahe(false);
        setSharpen(true);
      } else if (e.key.toLowerCase() === "p") {
        setPixelated((v) => !v);
      } else if (e.key.toLowerCase() === "z") {
        setZoomMode((curr) => {
          if (curr === "fit") return "1x";
          if (curr === "1x") return "2x";
          if (curr === "2x") return "4x";
          return "fit";
        });
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space" || e.key === "\\") {
        e.preventDefault();
        setIsComparing(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [onClose]);

  // Compute active image URL
  let currentImageUrl = src;
  if (detectionId) {
    if (isComparing) {
      currentImageUrl = getCropVariantUrl(detectionId, { preset: "raw", source });
    } else {
      currentImageUrl = getCropVariantUrl(detectionId, { chroma, clahe, sharpen, source });
    }
  }

  const applyPreset = (p: PresetId) => {
    if (p === "polish") {
      setChroma(true);
      setClahe(true);
      setSharpen(true);
    } else if (p === "raw") {
      setChroma(false);
      setClahe(false);
      setSharpen(false);
    } else if (p === "chroma_only") {
      setChroma(true);
      setClahe(false);
      setSharpen(false);
    } else if (p === "clahe_only") {
      setChroma(false);
      setClahe(true);
      setSharpen(false);
    } else if (p === "sharpen_only") {
      setChroma(false);
      setClahe(false);
      setSharpen(true);
    }
  };

  // Compute rendered dimensions
  const getRenderStyle = () => {
    if (!naturalDims) {
      return { maxWidth: "100%", maxHeight: "100%" };
    }
    const pxStyle = pixelated && zoomMode !== "fit" ? { imageRendering: "pixelated" as const } : {};
    if (zoomMode === "1x") {
      return { width: naturalDims.w, height: naturalDims.h, ...pxStyle };
    }
    if (zoomMode === "2x") {
      return { width: naturalDims.w * 2, height: naturalDims.h * 2, ...pxStyle };
    }
    if (zoomMode === "4x") {
      return { width: naturalDims.w * 4, height: naturalDims.h * 4, ...pxStyle };
    }
    // "fit" mode
    const scale = Math.min(
      (window.innerWidth * 0.90) / naturalDims.w,
      (window.innerHeight * 0.70) / naturalDims.h,
    );
    return {
      width: Math.round(naturalDims.w * scale),
      height: Math.round(naturalDims.h * scale),
      ...pxStyle,
    };
  };

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex flex-col justify-between p-3 sm:p-5 backdrop-blur-md bg-[color-mix(in_oklab,#090b08_94%,transparent)] select-none"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Computational Photography Inspector"
    >
      {/* Top Header & Preset Bar */}
      <div
        className="flex flex-wrap items-center justify-between gap-3 p-2.5 rounded-xl bg-surface/85 border border-line backdrop-blur-md z-10"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Left: Metadata */}
        <div className="flex items-center gap-3">
          <div>
            <div className="font-semibold text-sm text-ink leading-tight">
              {species ?? alt}
            </div>
            <div className="flex items-center gap-2 text-xs text-muted">
              {naturalDims && (
                <span>
                  {naturalDims.w} × {naturalDims.h} px
                </span>
              )}
              {initialSharpness != null && (
                <span title="Laplacian sharpness variance">
                  · var {Math.round(initialSharpness)}
                </span>
              )}
              {brightness != null && (
                <span title="Mean luminance">· bright {Math.round(brightness)}</span>
              )}
            </div>
          </div>
        </div>

        {/* Center: Presets */}
        {detectionId && (
          <div className="flex items-center gap-1 p-1 rounded-lg bg-panel/80 border border-line">
            <button
              className={`px-2.5 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "polish"
                  ? "bg-leaf text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("polish")}
              title="All enhancements active (production default) [Press 1]"
            >
              Full Polish
            </button>
            <button
              className={`px-2.5 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "raw"
                  ? "bg-amber-600/90 text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("raw")}
              title="Camera raw sensor pixels (no processing) [Press 2]"
            >
              Raw
            </button>
            <button
              className={`px-2 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "chroma_only"
                  ? "bg-accent text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("chroma_only")}
              title="Chroma-guided 4:2:0 filter only [Press 3]"
            >
              Chroma Only
            </button>
            <button
              className={`px-2 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "clahe_only"
                  ? "bg-accent text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("clahe_only")}
              title="Dynamic range CLAHE only [Press 4]"
            >
              CLAHE Only
            </button>
            <button
              className={`px-2 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "sharpen_only"
                  ? "bg-accent text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("sharpen_only")}
              title="Bilateral unsharp mask only [Press 5]"
            >
              Sharpen Only
            </button>
          </div>
        )}

        {/* Right: Zoom & Close Controls */}
        <div className="flex items-center gap-1.5">
          <div className="flex items-center p-0.5 rounded-lg bg-panel/80 border border-line text-xs">
            {(["fit", "1x", "2x", "4x"] as ZoomMode[]).map((mode) => (
              <button
                key={mode}
                className={`px-2 py-1 rounded transition-colors ${
                  zoomMode === mode
                    ? "bg-surface text-ink font-semibold shadow-xs"
                    : "text-muted hover:text-ink"
                }`}
                onClick={() => setZoomMode(mode)}
              >
                {mode === "fit" ? "Fit" : mode}
              </button>
            ))}
          </div>

          <button
            className={`px-2 py-1 text-xs rounded-lg border border-line transition-colors ${
              pixelated
                ? "bg-leaf/20 text-leaf border-leaf/40 font-medium"
                : "bg-panel text-muted hover:text-ink"
            }`}
            onClick={() => setPixelated((v) => !v)}
            title="Nearest-neighbor crisp pixel rendering [Press P]"
          >
            Pixels
          </button>

          <button
            className="grid place-items-center w-8 h-8 rounded-lg text-white/80 hover:text-white bg-white/10 hover:bg-white/20 transition-colors ml-1"
            onClick={onClose}
            aria-label="Close preview"
          >
            <CloseIcon size={16} />
          </button>
        </div>
      </div>

      {/* Main Image Viewport */}
      <div className="flex-1 overflow-auto flex items-center justify-center p-2 relative my-2">
        <div className="relative inline-block">
          <img
            src={currentImageUrl}
            alt={alt}
            className="object-contain rounded-card shadow-2xl transition-all"
            style={getRenderStyle()}
            onLoad={(e) => {
              const img = e.currentTarget;
              if (img.naturalWidth && img.naturalHeight) {
                setNaturalDims({ w: img.naturalWidth, h: img.naturalHeight });
              }
            }}
            onClick={(e) => e.stopPropagation()}
            draggable={false}
          />

          {/* Active status pill */}
          <div className="absolute top-2 left-2 z-10 pointer-events-none">
            {isComparing ? (
              <span className="px-2.5 py-1 text-xs font-bold uppercase tracking-wider rounded-md bg-amber-600 text-white shadow-lg animate-pulse">
                Before (Raw)
              </span>
            ) : activePreset === "polish" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-black/60 text-white/90 backdrop-blur-md border border-white/10">
                Full Polish
              </span>
            ) : activePreset === "raw" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-amber-900/60 text-amber-200 backdrop-blur-md border border-amber-500/20">
                Raw Camera Crop
              </span>
            ) : (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-black/60 text-white/90 backdrop-blur-md border border-white/10">
                {[
                  chroma && "Chroma",
                  clahe && "CLAHE",
                  sharpen && "Sharpen",
                ]
                  .filter(Boolean)
                  .join(" + ")}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Technique Toggles & A/B Compare Bar */}
      {detectionId && (
        <div
          className="flex flex-wrap items-center justify-between gap-3 p-2.5 rounded-xl bg-surface/85 border border-line backdrop-blur-md z-10"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Individual Technique Toggles */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted font-medium mr-1 hidden sm:inline">Techniques:</span>

            {/* Chroma-guided Denoising */}
            <button
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all ${
                chroma
                  ? "bg-leaf/15 border-leaf/50 text-ink font-semibold"
                  : "bg-panel border-line text-muted line-through opacity-70"
              }`}
              onClick={() => setChroma((v) => !v)}
              title="YCrCb guided filter: eliminates 4:2:0 subsampling bleed and chroma noise"
            >
              <i className={`w-2 h-2 rounded-full ${chroma ? "bg-leaf" : "bg-muted"}`} />
              Chroma Denoise
            </button>

            {/* CLAHE Lighting Normalization */}
            <button
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all ${
                clahe
                  ? "bg-leaf/15 border-leaf/50 text-ink font-semibold"
                  : "bg-panel border-line text-muted line-through opacity-70"
              }`}
              onClick={() => setClahe((v) => !v)}
              title="L-channel CLAHE: dynamic range shadow recovery without blowing highlights"
            >
              <i className={`w-2 h-2 rounded-full ${clahe ? "bg-leaf" : "bg-muted"}`} />
              Dynamic Range (CLAHE)
            </button>

            {/* Bilateral Unsharp Mask */}
            <button
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all ${
                sharpen
                  ? "bg-leaf/15 border-leaf/50 text-ink font-semibold"
                  : "bg-panel border-line text-muted line-through opacity-70"
              }`}
              onClick={() => setSharpen((v) => !v)}
              title="Adaptive edge-preserving bilateral sharpening: highlights plumage barbules without halos"
            >
              <i className={`w-2 h-2 rounded-full ${sharpen ? "bg-leaf" : "bg-muted"}`} />
              Plumage Sharpen
            </button>

            {/* Lucky Imaging Source Toggle (if available) */}
            {hasInitial && (
              <div className="flex items-center p-0.5 rounded-lg bg-panel border border-line ml-1">
                <button
                  className={`px-2 py-1 rounded transition-colors ${
                    source === "lucky"
                      ? "bg-leaf text-white font-medium"
                      : "text-muted hover:text-ink"
                  }`}
                  onClick={() => setSource("lucky")}
                  title="Sharpest micro-pause frame found via lucky imaging"
                >
                  Lucky Frame ★
                </button>
                <button
                  className={`px-2 py-1 rounded transition-colors ${
                    source === "initial"
                      ? "bg-leaf text-white font-medium"
                      : "text-muted hover:text-ink"
                  }`}
                  onClick={() => setSource("initial")}
                  title="Original 3fps sampled video frame before hunting"
                >
                  3fps Original
                </button>
              </div>
            )}
          </div>

          {/* Interactive Hold-to-Compare Button */}
          <div className="flex items-center gap-2">
            <button
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg border transition-all cursor-pointer ${
                isComparing
                  ? "bg-amber-600 text-white border-amber-500 shadow-md scale-95"
                  : "bg-surface hover:bg-panel text-ink border-line shadow-xs"
              }`}
              onMouseDown={() => setIsComparing(true)}
              onMouseUp={() => setIsComparing(false)}
              onMouseLeave={() => setIsComparing(false)}
              onTouchStart={() => setIsComparing(true)}
              onTouchEnd={() => setIsComparing(false)}
              title="Hold to temporarily show the raw unprocessed camera frame"
            >
              {isComparing ? "Showing Raw…" : "Hold to Compare (Spacebar)"}
            </button>
          </div>
        </div>
      )}
    </div>,
    document.body,
  );
}
