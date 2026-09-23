import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchFeederBehavior, type FeederBehaviorResponse } from "../lib/api";

export function FeederScience() {
  const [dwellTab, setDwellTab] = useState<"quick" | "sitters">("quick");
  const [pairSpeciesFilter, setPairSpeciesFilter] = useState<string>("all");

  const { data, isLoading, error } = useQuery<FeederBehaviorResponse>({
    queryKey: ["feeder-behavior"],
    queryFn: fetchFeederBehavior,
    refetchInterval: 5 * 60 * 1000,
  });

  if (isLoading) {
    return (
      <div className="fg-card p-4 text-muted text-sm animate-pulse">
        Loading feeder behavioral science…
      </div>
    );
  }

  if (error || !data) {
    return null;
  }

  const { dimorphic_species, dwell_rankings, pair_highlights } = data;

  const reliableRankings = (() => {
    const filtered = dwell_rankings.filter((r) => r.sample_count >= 15);
    return filtered.length >= 6 ? filtered : dwell_rankings;
  })();

  const displayedRankings =
    dwellTab === "quick"
      ? [...reliableRankings].sort((a, b) => a.avg_seconds - b.avg_seconds).slice(0, 7)
      : [...reliableRankings].sort((a, b) => b.avg_seconds - a.avg_seconds).slice(0, 7);

  const pairSpeciesList = (() => {
    const counts = new Map<string, number>();
    for (const p of pair_highlights) {
      counts.set(p.species, (counts.get(p.species) || 0) + 1);
    }
    return Array.from(counts.entries()).map(([species, count]) => ({ species, count }));
  })();

  const displayedPairs = (() => {
    if (pairSpeciesFilter === "all") {
      return pair_highlights.slice(0, 12);
    }
    return pair_highlights.filter((p) => p.species === pairSpeciesFilter);
  })();

  return (
    <div className="space-y-4 pt-2 border-t border-line/60">
      <div className="mb-1">
        <div className="fg-overline text-leaf">Behavioral science</div>
        <h2 className="font-serif font-medium text-2xl text-ink leading-tight mt-0.5">
          Feeder Science & Plumage Dynamics
        </h2>
        <p className="text-xs text-muted mt-1">
          Computer vision plumage analysis of sexual dimorphism, mated pair dynamics, and feeder dwell durations.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {/* Plumage Dimorphism & Pair Dynamics */}
        <section className="fg-card p-4 space-y-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h3 className="font-serif text-[17px] font-medium text-ink">
                Plumage Dimorphism
              </h3>
              <span className="text-[11px] text-faint">♂ Male vs. ♀ Female</span>
            </div>
            <p className="text-xs text-muted mt-0.5">
              Sex ratios computed from plumage color metrics and mask distributions.
            </p>

            <div className="mt-3 space-y-3">
              {dimorphic_species.map((sp) => {
                const total = sp.male + sp.female;
                return (
                  <div key={sp.species} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-ink">{sp.species}</span>
                      <div className="flex items-center gap-2">
                        {sp.pair_visits > 0 && (
                          <span
                            className="inline-flex items-center gap-0.5 text-[10px] font-semibold text-amber-700 dark:text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded px-1.5 py-0.5"
                            title={`${sp.pair_visits} visits with both male and female together`}
                          >
                            ❤️ {sp.pair_visits} pair visit{sp.pair_visits === 1 ? "" : "s"}
                          </span>
                        )}
                        <span className="text-faint tnum text-[11px]">
                          {total} classified
                        </span>
                      </div>
                    </div>

                    {total > 0 ? (
                      <div className="space-y-0.5">
                        <div className="h-2 w-full bg-surface rounded-full overflow-hidden flex border border-line/40">
                          <div
                            style={{ width: `${sp.male_pct}%` }}
                            className="h-full bg-sky-500/80 transition-all duration-500"
                            title={`Male: ${sp.male} (${sp.male_pct}%)`}
                          />
                          <div
                            style={{ width: `${sp.female_pct}%` }}
                            className="h-full bg-rose-500/80 transition-all duration-500"
                            title={`Female: ${sp.female} (${sp.female_pct}%)`}
                          />
                        </div>
                        <div className="flex justify-between text-[10px] text-muted tnum px-0.5">
                          <span className="text-sky-700 dark:text-sky-300 font-medium">
                            ♂ {sp.male_pct}% ({sp.male})
                          </span>
                          <span className="text-rose-700 dark:text-rose-300 font-medium">
                            ♀ {sp.female_pct}% ({sp.female})
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div className="text-[11px] text-faint italic py-1">
                        Awaiting clear plumage crops
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="pt-2 border-t border-line/40 text-[11px] text-faint">
            Classifies scarlet vs. olive in Cardinals, red bibs in Finches, black bibs in House Sparrows, and red nape spots in Woodpeckers.
          </div>
        </section>

        {/* Feeder Dwell Time Rankings */}
        <section className="fg-card p-4 space-y-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h3 className="font-serif text-[17px] font-medium text-ink">
                Feeder Dwell Durations
              </h3>
              <div className="inline-flex rounded-lg border border-line/60 bg-surface/80 p-0.5 text-[11px]">
                <button
                  type="button"
                  onClick={() => setDwellTab("quick")}
                  className={`px-2 py-0.5 rounded-md font-medium transition-colors ${
                    dwellTab === "quick"
                      ? "bg-leaf/20 text-leaf font-semibold shadow-xs"
                      : "text-muted hover:text-ink"
                  }`}
                >
                  ⚡ Quickest
                </button>
                <button
                  type="button"
                  onClick={() => setDwellTab("sitters")}
                  className={`px-2 py-0.5 rounded-md font-medium transition-colors ${
                    dwellTab === "sitters"
                      ? "bg-purple-600/20 text-purple-700 dark:text-purple-300 font-semibold shadow-xs"
                      : "text-muted hover:text-ink"
                  }`}
                >
                  🛋️ Longest
                </button>
              </div>
            </div>
            <p className="text-xs text-muted mt-0.5">
              {dwellTab === "quick"
                ? "Fastest grab-and-go foragers (average seconds per visit, ≥15 sightings)."
                : "Top lingering tray sitters (longest average visit, ≥15 sightings)."}
            </p>

            <div className="mt-3 space-y-2">
              {displayedRankings.map((d) => {
                const maxDwell = Math.max(...reliableRankings.map((r) => r.avg_seconds), 3.0);
                const pct = Math.min((d.avg_seconds / maxDwell) * 100, 100);
                const styleBadgeColor =
                  d.style === "Quick Forager"
                    ? "bg-leaf/10 text-leaf border-leaf/25"
                    : d.style === "Active Feeder"
                      ? "bg-amber-500/10 text-amber-800 dark:text-amber-300 border-amber-500/25"
                      : "bg-purple-600/10 text-purple-800 dark:text-purple-300 border-purple-500/25";

                const barColor =
                  d.style === "Quick Forager"
                    ? "bg-leaf/70"
                    : d.style === "Active Feeder"
                      ? "bg-amber-500/70"
                      : "bg-purple-600/70 dark:bg-purple-400/70";

                return (
                  <div key={d.species} className="space-y-0.5">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5 truncate pr-2">
                        <span className="font-medium text-ink truncate">{d.species}</span>
                        <span
                          className={`text-[9px] uppercase tracking-wider font-semibold px-1 py-0.2 rounded border ${styleBadgeColor} shrink-0`}
                        >
                          {d.style}
                        </span>
                        <span className="text-[10px] text-faint shrink-0">
                          ({d.sample_count} visits)
                        </span>
                      </div>
                      <span className="font-semibold text-ink tnum shrink-0">
                        {d.avg_seconds}s
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-surface rounded-full overflow-hidden border border-line/40">
                      <div
                        style={{ width: `${Math.max(pct, 5)}%` }}
                        className={`h-full ${barColor} rounded-full transition-all duration-500`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="pt-2 border-t border-line/40 text-[11px] text-faint flex justify-between">
            <span>Quick (&lt;1.2s) · Active (1.2–2.0s) · Sitter (&gt;2.0s)</span>
            <span className="tnum">{reliableRankings.length} regular species</span>
          </div>
        </section>
      </div>

      {/* Recent Mated Pair Visits */}
      {pair_highlights.length > 0 && (
        <section className="fg-card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-serif text-[17px] font-medium text-ink flex items-center gap-1.5">
                <span>❤️</span> Recent Mated Pair Sightings
              </h3>
              <p className="text-xs text-muted mt-0.5">
                Visits where both male and female of the same species fed together at the station.
              </p>
            </div>
            <span className="text-xs text-faint tnum">
              {displayedPairs.length} of {pair_highlights.length} events
            </span>
          </div>

          {pairSpeciesList.length > 1 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                onClick={() => setPairSpeciesFilter("all")}
                className={`px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors border ${
                  pairSpeciesFilter === "all"
                    ? "bg-leaf/20 text-leaf border-leaf/40 font-semibold shadow-xs"
                    : "border-line/60 bg-surface/50 text-muted hover:text-ink hover:border-line"
                }`}
              >
                All ({pair_highlights.length})
              </button>
              {pairSpeciesList.map(({ species, count }) => (
                <button
                  key={species}
                  type="button"
                  onClick={() => setPairSpeciesFilter(species)}
                  className={`px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors border ${
                    pairSpeciesFilter === species
                      ? "bg-leaf/20 text-leaf border-leaf/40 font-semibold shadow-xs"
                      : "border-line/60 bg-surface/50 text-muted hover:text-ink hover:border-line"
                  }`}
                >
                  {species} ({count})
                </button>
              ))}
            </div>
          )}

          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
            {displayedPairs.map((pair, idx) => (
              <div
                key={`${pair.visit_id}-${idx}`}
                className="rounded-lg border border-line/80 bg-surface/60 p-2.5 space-y-2 hover:border-leaf/50 transition-colors"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-serif font-medium text-ink">{pair.species} Pair</span>
                  <span className="text-[11px] text-faint">
                    {pair.started_at ? new Date(pair.started_at).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "Recent"}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-1.5">
                  {pair.crops.map((crop) => (
                    <div
                      key={crop.detection_id}
                      className="relative aspect-square rounded overflow-hidden bg-panel border border-line/60"
                    >
                      <img
                        src={crop.crop_url}
                        alt={`${pair.species} ${crop.sex}`}
                        className="w-full h-full object-contain"
                        loading="lazy"
                      />
                      <span
                        className={`absolute bottom-1 left-1 px-1 py-0.2 rounded text-[9px] font-semibold leading-none ${
                          crop.sex === "male"
                            ? "bg-sky-500/85 text-white"
                            : "bg-rose-500/85 text-white"
                        }`}
                      >
                        {crop.sex === "male" ? "♂ Male" : "♀ Female"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
