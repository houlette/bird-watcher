import type { Detection } from "../lib/api";
import AudioBadge from "./AudioBadge";

export default function DetectionCard({
  detection,
  compact = false,
}: {
  detection: Detection;
  compact?: boolean;
}) {
  const time = new Date(detection.created_at).toLocaleString();
  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden flex flex-col">
      <img
        src={detection.crop_url}
        alt={detection.species ?? "bird"}
        className="w-full object-cover aspect-[4/3]"
        loading="lazy"
      />
      <div className={`p-2 ${compact ? "text-xs" : "text-sm"}`}>
        <div className="flex items-center justify-between gap-2">
          <span className="font-semibold">{detection.species ?? "Unidentified"}</span>
          {detection.audio_confirmed && <AudioBadge />}
        </div>
        <div className="text-slate-500">{Math.round(detection.confidence * 100)}% · {time}</div>
      </div>
    </div>
  );
}
