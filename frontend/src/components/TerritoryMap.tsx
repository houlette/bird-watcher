import { useMemo, useState } from "react";

import type { TerritoryZone } from "../lib/api";

/**
 * The yard, with each zone shaded by whoever held it.
 *
 * SVG over the real camera still rather than a canvas, unlike the Art,
 * Tavern and Biome surfaces. Nothing here animates, the shapes are four
 * ellipses, and drawing them as elements means hit testing, focus rings
 * and screen-reader labels come free instead of being reimplemented
 * against a bitmap.
 *
 * The still is the same frame the location heatmap is drawn over, served
 * from the media mount. If it is missing the map still works: the zones
 * sit on a plain ground and the caller is told, because a zone map with
 * no yard behind it is a diagram of nothing in particular.
 */

export interface TerritoryMapProps {
  zones: TerritoryZone[];
  /** Which measure decided each holder, passed through for the labels. */
  basis: "seconds" | "appearances";
  selected: string | null;
  onSelect: (zone: TerritoryZone | null) => void;
}

/** 16:9, matching the 4K frame the bboxes are normalised against. */
const VB_W = 1000;
const VB_H = 563;

/** Where the camera still lives. Same mount the crops are served from. */
const YARD_STILL = "/media/calibration/heatmap_bg.jpg";

export function TerritoryMap({ zones, basis, selected, onSelect }: TerritoryMapProps) {
  const [stillFailed, setStillFailed] = useState(false);

  // Biggest first, so a small zone drawn inside a large one stays
  // clickable rather than being buried under it.
  const ordered = useMemo(
    () => [...zones].sort((a, b) => b.radius - a.radius),
    [zones]
  );

  return (
    <div className="relative w-full rounded-card overflow-hidden border border-line bg-panel">
      {stillFailed ? (
        <div className="w-full aspect-video bg-[color-mix(in_oklab,var(--accent)_8%,var(--panel))]" />
      ) : (
        <img
          src={YARD_STILL}
          alt="The feeder, seen from the camera"
          className="block w-full aspect-video object-cover"
          onError={() => setStillFailed(true)}
        />
      )}

      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="absolute inset-0 w-full h-full"
        role="group"
        aria-label="Feeder zones and who held them"
      >
        {/* A wash over the photo so the zone colours read against a busy
            garden. Without it the greens of the ivy fight every faction
            colour on the palette. */}
        <rect x={0} y={0} width={VB_W} height={VB_H} fill="rgba(12, 16, 12, 0.34)" />

        {ordered.map((zone) => {
          const held = zone.control[0];
          const isSelected = selected === zone.id;
          const color = held?.color ?? "#8a9285";
          const cx = zone.x * VB_W;
          const cy = zone.y * VB_H;
          const rx = zone.radius * VB_W;
          const ry = zone.radius * VB_H;
          // Fill tracks how decisively the zone was held, so a 50/50
          // stalemate looks faint and a rout looks solid.
          const fill = held ? 0.12 + 0.4 * held.share : 0.05;

          return (
            <g
              key={zone.id}
              onClick={() => onSelect(isSelected ? null : zone)}
              className="cursor-pointer"
              tabIndex={0}
              role="button"
              aria-label={`${zone.name}: ${
                held ? `${held.name}, ${Math.round(held.share * 100)}%` : "unheld"
              }`}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(isSelected ? null : zone);
                }
              }}
            >
              <ellipse
                cx={cx}
                cy={cy}
                rx={rx}
                ry={ry}
                fill={color}
                fillOpacity={fill}
                stroke={color}
                strokeOpacity={isSelected ? 1 : 0.85}
                strokeWidth={isSelected ? 4 : 2}
                strokeDasharray={zone.contested ? "10 7" : undefined}
              />

              <text
                x={cx}
                y={cy - 6}
                textAnchor="middle"
                className="fill-white"
                style={{ fontSize: 17, fontWeight: 600, paintOrder: "stroke" }}
                stroke="rgba(0,0,0,0.65)"
                strokeWidth={3.5}
              >
                {zone.name}
              </text>
              <text
                x={cx}
                y={cy + 14}
                textAnchor="middle"
                className="fill-white"
                style={{ fontSize: 14, paintOrder: "stroke" }}
                stroke="rgba(0,0,0,0.65)"
                strokeWidth={3}
              >
                {held
                  ? `${held.name} · ${Math.round(held.share * 100)}%`
                  : "nobody held this"}
              </text>
              {held && (
                <text
                  x={cx}
                  y={cy + 31}
                  textAnchor="middle"
                  className="fill-white/80"
                  style={{ fontSize: 12, paintOrder: "stroke" }}
                  stroke="rgba(0,0,0,0.6)"
                  strokeWidth={3}
                >
                  {basis === "seconds"
                    ? `${formatSeconds(zone.seconds)} held`
                    : `${zone.visits} landing${zone.visits === 1 ? "" : "s"}`}
                  {zone.contested ? " · contested" : ""}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {stillFailed && (
        <p className="absolute bottom-2 left-3 text-[11px] text-muted">
          The camera still is not on this server, so the zones are shown on their own.
        </p>
      )}
    </div>
  );
}

export function formatSeconds(seconds: number): string {
  if (seconds <= 0) return "0s";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s ? `${m}m ${s}s` : `${m}m`;
}
