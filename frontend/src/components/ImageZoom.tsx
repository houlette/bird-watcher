import { useEffect } from "react";

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
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
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
        className="max-w-full max-h-full object-contain rounded-card"
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
    </div>
  );
}
