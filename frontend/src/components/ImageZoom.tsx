import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { CloseIcon } from "./FieldIcons";

type Props = {
  src: string;
  alt: string;
  onClose: () => void;
};

/**
 * Full-screen image overlay for inspecting a crop more closely. Tap the
 * backdrop or hit Escape to dismiss. On mobile the native browser
 * pinch-zoom works on the <img> for finer inspection.
 *
 * Anchored at z-50 so it sits above the sticky toolbar and the bottom
 * action bars. The scrim is a deep ink wash (not pure black) so it reads
 * as part of the field-guide palette rather than a system dialog.
 */
export default function ImageZoom({ src, alt, onClose }: Props) {
  // Scale-to-fit, computed from the natural size once the image loads.
  // Feeder crops are often tiny (sub-150 px); max-w/max-h alone never
  // upscales, so the "zoom" used to show the crop at thumbnail size.
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Portal so the fixed-position overlay lays out against the viewport
  // instead of the parent DetectionCard, which has a transform-on-hover
  // (`.fg-liftable`) that creates a containing block for fixed
  // descendants and traps the overlay inside the card's bounds.
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 cursor-zoom-out backdrop-blur-sm bg-[color-mix(in_oklab,#0f110c_88%,transparent)]"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Image preview"
    >
      <img
        src={src}
        alt={alt}
        className="object-contain rounded-card"
        style={dims ? { width: dims.w, height: dims.h } : { maxWidth: "100%", maxHeight: "100%" }}
        onLoad={(e) => {
          const img = e.currentTarget;
          if (!img.naturalWidth || !img.naturalHeight) return;
          const scale = Math.min(
            (window.innerWidth * 0.92) / img.naturalWidth,
            (window.innerHeight * 0.85) / img.naturalHeight,
          );
          setDims({
            w: Math.round(img.naturalWidth * scale),
            h: Math.round(img.naturalHeight * scale),
          });
        }}
        onClick={(e) => e.stopPropagation()}
        draggable={false}
      />
      <button
        className="absolute top-4 right-4 grid place-items-center w-10 h-10 rounded-full text-white/80 hover:text-white bg-white/10 hover:bg-white/20 transition-colors"
        onClick={onClose}
        aria-label="Close preview"
      >
        <CloseIcon size={20} />
      </button>
    </div>,
    document.body,
  );
}
