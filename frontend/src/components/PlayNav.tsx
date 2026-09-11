import { NavLink, Outlet } from "react-router-dom";

import { FlagonIcon, PaletteIcon, ShieldIcon, SproutIcon } from "./FieldIcons";

/**
 * The four surfaces that read the archive a second way, behind one tab.
 *
 * They used to sit in the bottom nav individually, which worked at four
 * tabs and stopped working at nine: the labels needed about 530px and a
 * phone has 375, so the row folded in two. Grouping them puts the bottom
 * nav back to six tabs on one row and costs one tap to reach a surface
 * you were not already on.
 *
 * The routes are deliberately unchanged. `/art`, `/tavern`, `/biome` and
 * `/territory` are what the pages have always been at, so anything
 * bookmarked or pinned to a home screen still resolves. The grouping is
 * a layout wrapper, not a move.
 */

export interface PlaySurface {
  to: string;
  label: string;
  /** What the surface is, for the tooltip. */
  title: string;
  icon: React.ReactNode;
}

export const PLAY_SURFACES: PlaySurface[] = [
  {
    to: "/art",
    label: "Art",
    title: "Flightlines: a day of tracks as generative artwork",
    icon: <PaletteIcon size={15} />,
  },
  {
    to: "/tavern",
    label: "Tavern",
    title: "The Perch & Flagon: the feeder run as a tavern",
    icon: <FlagonIcon size={15} />,
  },
  {
    to: "/biome",
    label: "Biome",
    title: "Chrono-Chirps: the yard's audio grown as a garden",
    icon: <SproutIcon size={15} />,
  },
  {
    to: "/territory",
    label: "Wars",
    title: "Feeder Wars: the yard scored as ground held",
    icon: <ShieldIcon size={15} />,
  },
];

/** The paths the Play tab should light up for. */
export const PLAY_PATHS = PLAY_SURFACES.map((s) => s.to);

/** Where the Play tab lands when you have not chosen a surface yet. */
export const PLAY_HOME = PLAY_SURFACES[0].to;

export function PlayLayout() {
  return (
    <div className="space-y-4">
      {/* Scrolls rather than wraps if a future surface makes five: a
          sub-nav that reflows moves the page content under it. */}
      <div className="flex items-center gap-1 p-1 rounded-lg bg-panel border border-line w-fit max-w-full overflow-x-auto">
        {PLAY_SURFACES.map((s) => (
          <NavLink
            key={s.to}
            to={s.to}
            title={s.title}
            className={({ isActive }) =>
              `inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md whitespace-nowrap transition-colors ${
                isActive
                  ? "bg-surface text-leaf font-semibold"
                  : "text-muted hover:text-ink"
              }`
            }
          >
            {s.icon}
            {s.label}
          </NavLink>
        ))}
      </div>

      <Outlet />
    </div>
  );
}
