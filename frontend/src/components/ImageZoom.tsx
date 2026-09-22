import { useEffect, useRef, useState } from "react";
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
  hasLucky?: boolean;
  hasSr?: boolean;
  hasSisr?: boolean;
  onClose: () => void;
};

type PresetId =
  | "polish"
  | "bokeh"
  | "bokeh_only"
  | "mertens_hdr"
  | "raw"
  | "super_res"
  | "super_res_only"
  | "neural_sr"
  | "neural_sr_only"
  | "mertens_only"
  | "chroma_only"
  | "clahe_only"
  | "sharpen_only"
  | "custom";

type ZoomMode = "fit" | "1x" | "2x" | "4x";

/**
 * Full-screen computational photography inspection studio for crops.
 *
 * Allows toggling between combinations of techniques:
 *   - Multi-frame shift-and-add super-resolution: sub-pixel burst alignment for true 2x detail
 *   - Single-image neural super-resolution: 2x FSRCNN deep learning reconstruction
 *   - Chroma-guided filter (He et al.) in YCrCb: removes 4:2:0 subsampling bleed & color noise
 *   - Mertens multiscale exposure fusion: natural dynamic range recovery without halos
 *   - CLAHE: dynamic range lighting normalization on L channel
 *   - Bilateral unsharp masking: plumage edge sharpening without halos
 *   - Synthetic bokeh defocus: edge-preserving background optical blur
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
  cropAreaPx,
  brightness,
  hasLucky,
  hasSr,
  hasSisr,
  onClose,
}: Props) {
  // Technique toggle states
  const [sr, setSr] = useState(false);
  const [sisr, setSisr] = useState(false);
  const [chroma, setChroma] = useState(true);
  const [mertens, setMertens] = useState(true);
  const [clahe, setClahe] = useState(true);
  const [sharpen, setSharpen] = useState(true);
  const [bokeh, setBokeh] = useState(false);
  const [source, setSource] = useState<"lucky" | "initial">("lucky");
  const [hasInitial, setHasInitial] = useState(hasLucky ?? false);
  const [hasSrState, setHasSrState] = useState(hasSr ?? false);
  const [hasSisrState, setHasSisrState] = useState(hasSisr ?? false);

  // Comparison & View states
  const [isComparing, setIsComparing] = useState(false);
  const [zoomMode, setZoomMode] = useState<ZoomMode>("fit");
  const [pixelated, setPixelated] = useState(true);

  // Natural image dimensions
  const [naturalDims, setNaturalDims] = useState<{ w: number; h: number } | null>(null);

  // Determine active preset
  let activePreset: PresetId = "custom";
  if (sr && chroma && mertens && clahe && sharpen && !bokeh) activePreset = "super_res";
  else if (sr && !chroma && !mertens && !clahe && !sharpen && !bokeh) activePreset = "super_res_only";
  else if (sisr && chroma && mertens && clahe && sharpen && !bokeh) activePreset = "neural_sr";
  else if (sisr && !chroma && !mertens && !clahe && !sharpen && !bokeh) activePreset = "neural_sr_only";
  else if (!sr && !sisr && chroma && mertens && clahe && sharpen && bokeh) activePreset = "bokeh";
  else if (!sr && !sisr && !chroma && !mertens && !clahe && !sharpen && bokeh) activePreset = "bokeh_only";
  else if (!sr && !sisr && chroma && mertens && clahe && sharpen && !bokeh) activePreset = "polish";
  else if (!sr && !sisr && chroma && mertens && !clahe && sharpen && !bokeh) activePreset = "mertens_hdr";
  else if (!sr && !sisr && !chroma && !mertens && !clahe && !sharpen && !bokeh) activePreset = "raw";
  else if (!sr && !sisr && !chroma && mertens && !clahe && !sharpen && !bokeh) activePreset = "mertens_only";
  else if (!sr && !sisr && chroma && !mertens && !clahe && !sharpen && !bokeh) activePreset = "chroma_only";
  else if (!sr && !sisr && !chroma && !mertens && clahe && !sharpen && !bokeh) activePreset = "clahe_only";
  else if (!sr && !sisr && !chroma && !mertens && !clahe && sharpen && !bokeh) activePreset = "sharpen_only";

  // Pre-fetch metadata & standard presets into browser cache
  useEffect(() => {
    if (!detectionId) return;
    fetchCropVariantsMeta(detectionId)
      .then((meta) => {
        if (meta.has_initial) setHasInitial(true);
        if (meta.has_sr) setHasSrState(true);
        if (meta.has_sisr) setHasSisrState(true);
        // Preload standard preset images so toggling is 100% instantaneous
        ["polish", "bokeh", "mertens_hdr", "raw", "super_res", "neural_sr", "mertens_only", "chroma_only", "clahe_only", "sharpen_only", "bokeh_only"].forEach((p) => {
          const img = new Image();
          img.src = getCropVariantUrl(detectionId, { preset: p });
        });
      })
      .catch(() => {});
  }, [detectionId]);

  const modalRef = useRef<HTMLDivElement>(null);

  // Focus modal container on mount so activeElement is within the overlay
  useEffect(() => {
    modalRef.current?.focus();
  }, []);

  const applyPreset = (p: PresetId) => {
    if (p === "super_res") {
      setSr(true);
      setSisr(false);
      setChroma(true);
      setMertens(true);
      setClahe(true);
      setSharpen(true);
      setBokeh(false);
    } else if (p === "super_res_only") {
      setSr(true);
      setSisr(false);
      setChroma(false);
      setMertens(false);
      setClahe(false);
      setSharpen(false);
      setBokeh(false);
    } else if (p === "neural_sr") {
      setSr(false);
      setSisr(true);
      setChroma(true);
      setMertens(true);
      setClahe(true);
      setSharpen(true);
      setBokeh(false);
    } else if (p === "neural_sr_only") {
      setSr(false);
      setSisr(true);
      setChroma(false);
      setMertens(false);
      setClahe(false);
      setSharpen(false);
      setBokeh(false);
    } else if (p === "bokeh") {
      setSr(false);
      setSisr(false);
      setChroma(true);
      setMertens(true);
      setClahe(true);
      setSharpen(true);
      setBokeh(true);
    } else if (p === "bokeh_only") {
      setSr(false);
      setSisr(false);
      setChroma(false);
      setMertens(false);
      setClahe(false);
      setSharpen(false);
      setBokeh(true);
    } else if (p === "polish") {
      setSr(false);
      setSisr(false);
      setChroma(true);
      setMertens(true);
      setClahe(true);
      setSharpen(true);
      setBokeh(false);
    } else if (p === "mertens_hdr") {
      setSr(false);
      setSisr(false);
      setChroma(true);
      setMertens(true);
      setClahe(false);
      setSharpen(true);
      setBokeh(false);
    } else if (p === "raw") {
      setSr(false);
      setSisr(false);
      setChroma(false);
      setMertens(false);
      setClahe(false);
      setSharpen(false);
      setBokeh(false);
    } else if (p === "mertens_only") {
      setSr(false);
      setSisr(false);
      setChroma(false);
      setMertens(true);
      setClahe(false);
      setSharpen(false);
      setBokeh(false);
    } else if (p === "chroma_only") {
      setSr(false);
      setSisr(false);
      setChroma(true);
      setMertens(false);
      setClahe(false);
      setSharpen(false);
      setBokeh(false);
    } else if (p === "clahe_only") {
      setSr(false);
      setSisr(false);
      setChroma(false);
      setMertens(false);
      setClahe(true);
      setSharpen(false);
      setBokeh(false);
    } else if (p === "sharpen_only") {
      setSr(false);
      setSisr(false);
      setChroma(false);
      setMertens(false);
      setClahe(false);
      setSharpen(true);
      setBokeh(false);
    }
  };

  // Keyboard navigation & spacebar compare
  useEffect(() => {
    const isSpaceKey = (e: KeyboardEvent) =>
      e.code === "Space" ||
      e.key === " " ||
      e.key === "Spacebar" ||
      e.keyCode === 32 ||
      e.which === 32;

    const isBackslash = (e: KeyboardEvent) =>
      e.key === "\\" ||
      e.code === "Backslash" ||
      e.keyCode === 220 ||
      e.which === 220;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.keyCode === 27) {
        e.preventDefault();
        e.stopPropagation();
        onClose();
        return;
      }

      if (isSpaceKey(e) || isBackslash(e)) {
        // Unconditionally prevent default and stop propagation so spacebar
        // NEVER scrolls the page or triggers button clicks
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation?.();
        if (!e.repeat) {
          setIsComparing(true);
        }
        return;
      }

      if (e.key === "1") {
        applyPreset("polish");
      } else if (e.key === "2") {
        applyPreset("mertens_hdr");
      } else if (e.key === "3") {
        applyPreset("raw");
      } else if (e.key === "4") {
        applyPreset("super_res");
      } else if (e.key === "5") {
        applyPreset("mertens_only");
      } else if (e.key === "6") {
        applyPreset("chroma_only");
      } else if (e.key === "7") {
        applyPreset("sharpen_only");
      } else if (e.key === "8") {
        applyPreset("neural_sr");
      } else if (e.key === "9") {
        applyPreset("bokeh");
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
      if (isSpaceKey(e) || isBackslash(e)) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation?.();
        setIsComparing(false);
      }
    };

    // Attach in capture phase on both window and document to intercept before
    // browser default action, target elements, or scroll containers.
    window.addEventListener("keydown", onKeyDown, { capture: true });
    window.addEventListener("keyup", onKeyUp, { capture: true });
    document.addEventListener("keydown", onKeyDown, { capture: true });
    document.addEventListener("keyup", onKeyUp, { capture: true });

    return () => {
      window.removeEventListener("keydown", onKeyDown, { capture: true });
      window.removeEventListener("keyup", onKeyUp, { capture: true });
      document.removeEventListener("keydown", onKeyDown, { capture: true });
      document.removeEventListener("keyup", onKeyUp, { capture: true });
    };
  }, [onClose]);

  // Compute active image URL
  let currentImageUrl = src;
  if (detectionId) {
    if (isComparing) {
      currentImageUrl = getCropVariantUrl(detectionId, { preset: "raw", source });
    } else {
      currentImageUrl = getCropVariantUrl(detectionId, { chroma, mertens, clahe, sharpen, sr, sisr, bokeh, source });
    }
  }



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
      ref={modalRef}
      tabIndex={-1}
      className="fixed inset-0 z-50 flex flex-col justify-between p-3 sm:p-5 backdrop-blur-md bg-[color-mix(in_oklab,#090b08_94%,transparent)] select-none overscroll-contain outline-none"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Computational Photography Inspector"
      onWheel={(e) => {
        if (e.target === e.currentTarget) {
          e.preventDefault();
        }
      }}
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
              {cropAreaPx != null && (
                <span title="Original detection pixel area">
                  · {cropAreaPx.toLocaleString()} px²
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
              {hasInitial && (
                <span
                  className="rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 px-1.5 py-0.5 text-[9px] font-bold shadow-xs flex items-center gap-0.5 ml-1"
                  title="Lucky imaging found and swapped in a sharper video frame"
                >
                  ★ Lucky Frame
                </span>
              )}
              {hasSrState && (
                <span
                  className="rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30 px-1.5 py-0.5 text-[9px] font-bold shadow-xs flex items-center gap-0.5 ml-0.5"
                  title="2x Multi-frame shift-and-add super-resolution reconstruction available"
                >
                  2× SR
                </span>
              )}
              {hasSisrState && (
                <span
                  className="rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-1.5 py-0.5 text-[9px] font-bold shadow-xs flex items-center gap-0.5 ml-0.5"
                  title="2x Neural single-image super-resolution (FSRCNN) reconstruction available"
                >
                  2× Neural
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Center: Presets */}
        {detectionId && (
          <div className="flex flex-wrap items-center gap-1 p-1 rounded-lg bg-panel/80 border border-line">
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
                activePreset === "bokeh"
                  ? "bg-rose-600 text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("bokeh")}
              title="Full polish + synthetic optical bokeh background defocus [Press 9]"
            >
              Bokeh
            </button>
            <button
              className={`px-2.5 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "mertens_hdr"
                  ? "bg-teal-600 text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("mertens_hdr")}
              title="Mertens multiscale exposure fusion + Chroma + Sharpen (no CLAHE) [Press 2]"
            >
              Mertens HDR
            </button>
            <button
              className={`px-2.5 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "raw"
                  ? "bg-amber-600/90 text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("raw")}
              title="Camera raw sensor pixels (no processing) [Press 3]"
            >
              Raw
            </button>
            <button
              className={`px-2.5 py-1 text-xs rounded font-medium transition-all flex items-center gap-1 ${
                activePreset === "super_res"
                  ? "bg-purple-600 text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("super_res")}
              title="Multi-frame shift-and-add 2x super-resolution + full polish [Press 4]"
            >
              <span>Super-Res 2x</span>
              {hasSrState && (
                <span className="w-1.5 h-1.5 rounded-full bg-purple-400" title="Pre-generated" />
              )}
            </button>
            <button
              className={`px-2.5 py-1 text-xs rounded font-medium transition-all flex items-center gap-1 ${
                activePreset === "neural_sr"
                  ? "bg-indigo-600 text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("neural_sr")}
              title="Single-image neural 2x super-resolution (FSRCNN via OpenVINO) + full polish [Press 8]"
            >
              <span>Neural SR 2x</span>
              {hasSisrState && (
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" title="Pre-generated" />
              )}
            </button>
            <button
              className={`px-2 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "mertens_only"
                  ? "bg-accent text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("mertens_only")}
              title="Mertens multiscale exposure fusion only [Press 5]"
            >
              Mertens Only
            </button>
            <button
              className={`px-2 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "chroma_only"
                  ? "bg-accent text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("chroma_only")}
              title="Chroma-guided 4:2:0 filter only [Press 6]"
            >
              Chroma Only
            </button>
            <button
              className={`px-2 py-1 text-xs rounded font-medium transition-all ${
                activePreset === "sharpen_only"
                  ? "bg-accent text-white shadow-sm font-semibold"
                  : "text-muted hover:text-ink hover:bg-surface/60"
              }`}
              onClick={() => applyPreset("sharpen_only")}
              title="Bilateral unsharp mask only [Press 7]"
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
      <div className="flex-1 overflow-auto flex items-center justify-center p-2 relative my-2 overscroll-contain">
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
            ) : activePreset === "super_res" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-purple-900/80 text-purple-100 backdrop-blur-md border border-purple-500/30">
                Super-Res 2x (Shift-and-Add)
              </span>
            ) : activePreset === "super_res_only" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-purple-900/80 text-purple-100 backdrop-blur-md border border-purple-500/30">
                Super-Res 2x (Unprocessed)
              </span>
            ) : activePreset === "neural_sr" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-indigo-900/80 text-indigo-100 backdrop-blur-md border border-indigo-500/30">
                Neural SR 2x (FSRCNN OpenVINO)
              </span>
            ) : activePreset === "neural_sr_only" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-indigo-900/80 text-indigo-100 backdrop-blur-md border border-indigo-500/30">
                Neural SR 2x (Unprocessed)
              </span>
            ) : activePreset === "polish" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-black/60 text-white/90 backdrop-blur-md border border-white/10">
                Full Polish (Chroma + Mertens + CLAHE + Sharpen)
              </span>
            ) : activePreset === "bokeh" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-rose-900/80 text-rose-100 backdrop-blur-md border border-rose-500/30">
                Full Polish + Synthetic Bokeh Defocus
              </span>
            ) : activePreset === "bokeh_only" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-rose-900/80 text-rose-100 backdrop-blur-md border border-rose-500/30">
                Synthetic Bokeh Defocus Only
              </span>
            ) : activePreset === "mertens_hdr" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-teal-900/80 text-teal-100 backdrop-blur-md border border-teal-500/30">
                Mertens HDR Fusion (No Halos)
              </span>
            ) : activePreset === "raw" ? (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-amber-900/60 text-amber-200 backdrop-blur-md border border-amber-500/20">
                Raw Camera Crop
              </span>
            ) : (
              <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-black/60 text-white/90 backdrop-blur-md border border-white/10">
                {[
                  sr && "Super-Res 2x",
                  sisr && "Neural SR 2x",
                  chroma && "Chroma",
                  mertens && "Mertens HDR",
                  clahe && "CLAHE",
                  sharpen && "Sharpen",
                  bokeh && "Bokeh",
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

            {/* Multi-frame Shift-and-Add Super-Resolution */}
            <button
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all ${
                sr
                  ? "bg-purple-500/20 border-purple-500/50 text-ink font-semibold"
                  : "bg-panel border-line text-muted line-through opacity-70"
              }`}
              onClick={() => {
                setSr((v) => {
                  const next = !v;
                  if (next) setSisr(false);
                  return next;
                });
              }}
              title="Multi-frame shift-and-add: sub-pixel burst frame registration for optical resolution gain and noise reduction"
            >
              <i className={`w-2 h-2 rounded-full ${sr ? "bg-purple-500" : "bg-muted"}`} />
              Super-Res 2x
            </button>

            {/* Single-Image Neural Super-Resolution (FSRCNN) */}
            <button
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all ${
                sisr
                  ? "bg-indigo-500/20 border-indigo-500/50 text-ink font-semibold"
                  : "bg-panel border-line text-muted line-through opacity-70"
              }`}
              onClick={() => {
                setSisr((v) => {
                  const next = !v;
                  if (next) setSr(false);
                  return next;
                });
              }}
              title="Single-image neural super-resolution: 2x deep learning reconstruction via lightweight FSRCNN in OpenVINO"
            >
              <i className={`w-2 h-2 rounded-full ${sisr ? "bg-indigo-500" : "bg-muted"}`} />
              Neural SR 2x
            </button>

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

            {/* Mertens Multiscale Exposure Fusion */}
            <button
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all ${
                mertens
                  ? "bg-leaf/15 border-leaf/50 text-ink font-semibold"
                  : "bg-panel border-line text-muted line-through opacity-70"
              }`}
              onClick={() => setMertens((v) => !v)}
              title="Tommert & Mertens multiscale exposure fusion: blends synthetic brackets to lift shadows and preserve highlights with zero halos"
            >
              <i className={`w-2 h-2 rounded-full ${mertens ? "bg-leaf" : "bg-muted"}`} />
              Exposure Fusion (Mertens HDR)
            </button>

            {/* CLAHE Lighting Normalization */}
            <button
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all ${
                clahe
                  ? "bg-leaf/15 border-leaf/50 text-ink font-semibold"
                  : "bg-panel border-line text-muted line-through opacity-70"
              }`}
              onClick={() => setClahe((v) => !v)}
              title="L-channel CLAHE: local contrast dynamic range redistribution"
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

            {/* Synthetic Bokeh / Defocus */}
            <button
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-all ${
                bokeh
                  ? "bg-rose-500/20 border-rose-500/50 text-ink font-semibold"
                  : "bg-panel border-line text-muted line-through opacity-70"
              }`}
              onClick={() => setBokeh((v) => !v)}
              title="Synthetic Bokeh: edge-guided optical lens defocus blur isolating bird from background clutter"
            >
              <i className={`w-2 h-2 rounded-full ${bokeh ? "bg-rose-500" : "bg-muted"}`} />
              Bokeh Defocus
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
