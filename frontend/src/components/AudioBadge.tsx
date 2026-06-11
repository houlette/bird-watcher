import { SoundIcon } from "./FieldIcons";

/**
 * "Heard too" badge — the Haikubox confirmed this species by call at the
 * same time the camera saw it. A leaf-tinted pill with a small waveform
 * glyph, sized to sit unobtrusively beside the species name.
 */
export default function AudioBadge() {
  return (
    <span
      title="Haikubox heard this species at the same time"
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-leaf border border-[color-mix(in_oklab,var(--accent)_28%,var(--line))] bg-[color-mix(in_oklab,var(--accent)_12%,var(--card))]"
    >
      <SoundIcon size={11} />
      audio
    </span>
  );
}
