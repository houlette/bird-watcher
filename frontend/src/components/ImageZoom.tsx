import { useEffect } from "react";

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
 * action bars.
 */
export default function ImageZoom({ src, alt, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4 cursor-zoom-out"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Image preview"
    >
      <img
        src={src}
        alt={alt}
        className="max-w-full max-h-full object-contain"
        // Stop-propagation so tapping the image itself doesn't close —
        // user can drag/pinch-zoom on the img; closing requires the
        // backdrop or the close button.
        onClick={(e) => e.stopPropagation()}
        draggable={false}
      />
      <button
        className="absolute top-4 right-4 text-white/80 hover:text-white text-3xl leading-none"
        onClick={onClose}
        aria-label="Close preview"
      >
        ×
      </button>
    </div>
  );
}
