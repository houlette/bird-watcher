import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchDailyStory, type DailyStory } from "../lib/api";
import { ChevronIcon } from "./FieldIcons";

type Props = {
  targetDate?: string;
  selectedSpecies?: string;
  onSelectSpecies: (name: string | null) => void;
  defaultCollapsed?: boolean;
};

function formatStoryHeading(isoDate: string, isToday: boolean): string {
  if (isToday) return "Today's Yard Story";
  try {
    const [y, m, d] = isoDate.split("-").map(Number);
    const dateObj = new Date(y, m - 1, d);
    const now = new Date();
    const todayMidnight = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const diffDays = Math.round((todayMidnight.getTime() - dateObj.getTime()) / (24 * 3600 * 1000));
    const formatted = dateObj.toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
    if (diffDays === 1) {
      return `Yesterday's Yard Story · ${formatted}`;
    }
    return `Yard Story · ${formatted}`;
  } catch {
    return `Feeder Story · ${isoDate}`;
  }
}

export default function DailyStoryBulletin({
  targetDate,
  selectedSpecies,
  onSelectSpecies,
  defaultCollapsed = false,
}: Props) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  const { data: story, isLoading, error } = useQuery<DailyStory>({
    queryKey: ["daily_story", targetDate ?? "today"],
    queryFn: () => fetchDailyStory(targetDate),
    staleTime: targetDate ? 10 * 60_000 : 60_000,
    refetchInterval: targetDate ? false : 60_000,
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
    retry: 2,
  });

  if (isLoading || error || !story || !story.has_data || !story.hero) {
    return null;
  }

  const { hero, species_highlights: highlights } = story;
  const heroTime = new Date(hero.captured_at).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });

  return (
    <div className="fg-card p-4 md:p-5 mb-4 border border-[color-mix(in_oklab,var(--accent)_25%,var(--line))] bg-[color-mix(in_oklab,var(--panel)_50%,var(--card))]">
      {/* Header bar */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {story.is_today ? (
            <span className="fg-livedot" aria-hidden />
          ) : (
            <span className="inline-block w-2 h-2 rounded-full bg-leaf/40" aria-hidden />
          )}
          <span className="fg-overline">
            {formatStoryHeading(story.date, story.is_today)}
          </span>
          {!story.is_today && story.fallback_date && (
            <span className="text-[11px] text-muted italic">
              (latest recorded)
            </span>
          )}
        </div>
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="text-xs text-muted hover:text-ink flex items-center gap-1 font-medium transition-colors"
          aria-expanded={!collapsed}
          title={collapsed ? "Expand yard story" : "Collapse yard story"}
        >
          <span>{collapsed ? "Show details" : "Collapse"}</span>
          <ChevronIcon
            direction={collapsed ? "down" : "up"}
            className="w-3.5 h-3.5 opacity-70"
          />
        </button>
      </div>

      {/* Main summary line */}
      <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="font-serif font-medium text-xl md:text-2xl text-ink leading-tight">
          {story.species_count} species · {story.total_detections} sightings
        </h3>
        <span className="text-xs text-muted">
          across {story.total_visits} camera visit{story.total_visits === 1 ? "" : "s"}
        </span>
      </div>

      {!collapsed && (
        <div className="mt-4 pt-3.5 border-t border-line/60 grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
          {/* Hero Bird of the Day */}
          <div className="md:col-span-5 flex items-center gap-3.5 bg-surface/75 p-3 rounded-card border border-line shadow-sm">
            <div className="relative w-20 h-20 md:w-24 md:h-24 flex-shrink-0 rounded-lg overflow-hidden border border-line bg-panel">
              <img
                src={hero.crop_url}
                alt={hero.species}
                className="w-full h-full object-cover"
                loading="lazy"
              />
              <span className="absolute top-1 left-1 bg-surface/90 backdrop-blur-sm text-[9px] font-bold uppercase tracking-wider text-leaf px-1.5 py-0.5 rounded shadow-sm">
                Hero
              </span>
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-[11px] font-semibold text-faint tracking-wide uppercase">
                Hero of the Day
              </div>
              <button
                onClick={() => onSelectSpecies(hero.species)}
                className="text-left font-serif font-medium text-lg text-ink hover:text-leaf truncate block leading-snug"
                title={`Filter by ${hero.species}`}
              >
                {hero.species}
              </button>
              {hero.scientific_name && (
                <div className="text-xs italic text-muted truncate">
                  {hero.scientific_name}
                </div>
              )}
              <div className="mt-1.5 flex items-center gap-2 text-[11px] text-faint tnum">
                <span>{heroTime}</span>
                <span>·</span>
                <span className="font-medium text-leaf">
                  {Math.round(hero.confidence * 100)}% conf
                </span>
                {hero.sharpness && (
                  <>
                    <span>·</span>
                    <span>Q{Math.round(hero.sharpness)}</span>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Daily Diversity Strip */}
          <div className="md:col-span-7 flex flex-col justify-center min-w-0">
            <div className="flex items-center justify-between mb-2">
              <span className="fg-overline">
                Visitors Gallery ({highlights.length})
              </span>
              {selectedSpecies && (
                <button
                  onClick={() => onSelectSpecies(null)}
                  className="text-[11px] font-semibold text-leaf hover:underline"
                >
                  Clear filter
                </button>
              )}
            </div>

            {/* Horizontal scrolling strip */}
            <div className="flex items-center gap-2.5 overflow-x-auto pb-1 pt-0.5 no-scrollbar">
              {highlights.map((h) => {
                const isSelected = selectedSpecies === h.common_name;
                return (
                  <button
                    key={h.species_id}
                    onClick={() =>
                      onSelectSpecies(isSelected ? null : h.common_name)
                    }
                    className={`group flex-shrink-0 flex flex-col items-center p-1.5 rounded-lg border transition-all ${
                      isSelected
                        ? "bg-leaf/10 border-leaf shadow-sm"
                        : "bg-surface/60 border-line hover:border-leaf/60 hover:bg-surface"
                    }`}
                    title={`${h.common_name} (${h.count} sightings)`}
                  >
                    <div className="relative w-11 h-11 rounded-full overflow-hidden border border-line/80 group-hover:border-leaf transition-colors">
                      <img
                        src={h.best_crop_url}
                        alt={h.common_name}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    </div>
                    <span className="mt-1 text-[11px] font-medium text-ink max-w-[70px] truncate text-center leading-tight">
                      {h.common_name}
                    </span>
                    <span className="mt-0.5 px-1.5 py-0.2 rounded-full text-[9.5px] font-semibold bg-panel border border-line text-muted tnum">
                      {h.count}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
