import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";

import BulkActionBar from "../components/BulkActionBar";
import DailyStoryBulletin from "../components/DailyStoryBulletin";
import DetectionCard from "../components/DetectionCard";
import FilterPicker, { type Filter } from "../components/FilterPicker";
import { fetchDetections, type Detection } from "../lib/api";

const PAGE_SIZE = 50;

type Props = {
  // "feed": the everyday feed (hides NAB / Poor-quality).
  // "review": the maintenance surface — its filter offers the audit
  //   cohorts (past NAB labels, bad crops, binary-filter NABs) instead of
  //   the everyday browsing filters.
  surface?: "feed" | "review";
};

// "Best only" scoring: fused confidence first; tie-break on sharpness×area.
function bestScore(d: Detection): [number, number] {
  return [d.confidence, (d.sharpness ?? 0) * (d.crop_area_px ?? 0)];
}

function formatTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function formatDate(d: Date): string {
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function formatDuration(diffMinutes: number): string {
  if (diffMinutes < 60) return `${diffMinutes}m visit`;
  const hrs = Math.floor(diffMinutes / 60);
  const mins = diffMinutes % 60;
  return mins > 0 ? `${hrs}h ${mins}m visit` : `${hrs}h visit`;
}

function formatEncounterTime(first: Date, last: Date): string {
  const now = new Date();
  const isToday =
    first.getFullYear() === now.getFullYear() &&
    first.getMonth() === now.getMonth() &&
    first.getDate() === now.getDate();

  const diffMinutes = Math.round(Math.abs(last.getTime() - first.getTime()) / (60 * 1000));
  const tFirst = formatTime(first);
  const tLast = formatTime(last);
  const datePrefix = isToday ? "" : `${formatDate(first)} · `;

  if (diffMinutes < 1) {
    return `${datePrefix}${tLast}`;
  }
  return `${datePrefix}${tFirst} – ${tLast} (${formatDuration(diffMinutes)})`;
}

type EncounterCard = {
  det: Detection;
  count: number;
  label?: string;
  timeRange?: string;
  allCrops: Detection[];
};

// Encounter rollup: collapses sightings of the same species within a 15-minute
// sliding window into a single behavioral encounter card, regardless of interleaved
// sightings of other birds or unidentified crops.
function collapseEncounters(
  dets: Detection[],
  windowMinutes: number = 15,
): EncounterCard[] {
  if (dets.length === 0) return [];
  const windowMs = windowMinutes * 60 * 1000;

  const encounters: {
    best: Detection;
    count: number;
    firstTime: Date; // earliest timestamp in encounter
    lastTime: Date;  // latest timestamp in encounter
    speciesId: number | null;
    visitId?: number;
    allCrops: Detection[];
  }[] = [];

  const activeBySpecies = new Map<number, number>();
  const activeByVisit = new Map<number, number>();

  for (const d of dets) {
    const curTime = new Date(d.captured_at.endsWith("Z") ? d.captured_at : d.captured_at + "Z");
    const spId = d.species_id;

    let targetIdx: number | undefined;

    if (spId != null) {
      if (activeBySpecies.has(spId)) {
        const idx = activeBySpecies.get(spId)!;
        const enc = encounters[idx];
        const diff = Math.abs(enc.firstTime.getTime() - curTime.getTime());
        if (diff <= windowMs) {
          targetIdx = idx;
        } else {
          activeBySpecies.delete(spId);
        }
      }
    } else if (d.visit_id != null) {
      if (activeByVisit.has(d.visit_id)) {
        targetIdx = activeByVisit.get(d.visit_id)!;
      }
    }

    if (targetIdx !== undefined) {
      const enc = encounters[targetIdx];
      enc.count += 1;
      enc.allCrops.push(d);
      if (curTime < enc.firstTime) enc.firstTime = curTime;
      if (curTime > enc.lastTime) enc.lastTime = curTime;

      const [c1, q1] = bestScore(d);
      const [c0, q0] = bestScore(enc.best);
      if (c1 > c0 || (c1 === c0 && q1 > q0)) {
        enc.best = d;
      }
    } else {
      const newEnc = {
        best: d,
        count: 1,
        firstTime: curTime,
        lastTime: curTime,
        speciesId: spId,
        visitId: d.visit_id,
        allCrops: [d],
      };
      const newIdx = encounters.length;
      encounters.push(newEnc);
      if (spId != null) {
        activeBySpecies.set(spId, newIdx);
      } else if (d.visit_id != null) {
        activeByVisit.set(d.visit_id, newIdx);
      }
    }
  }

  return encounters.map((enc) => {
    // Put best shot first in thumbnail list
    const otherCrops = enc.allCrops.filter((c) => c.id !== enc.best.id);
    const sortedCrops = [enc.best, ...otherCrops];

    return {
      det: enc.best,
      count: enc.count,
      label: enc.count > 1 ? `${enc.count} in encounter` : undefined,
      timeRange: enc.count > 1 ? formatEncounterTime(enc.firstTime, enc.lastTime) : undefined,
      allCrops: sortedCrops,
    };
  });
}

export default function Feed({ surface = "feed" }: Props = {}) {
  const isReview = surface === "review";

  const [batchMode, setBatchMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const toggleSelect = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);
  const exitBatchMode = useCallback(() => {
    setSelectedIds(new Set());
    setBatchMode(false);
  }, []);

  // Persisted per surface: Feed unmounts on every tab switch, and losing a
  // species or review-queue filter on each round-trip made multi-page
  // review workflows painful. Feed and Review keep separate filters so they
  // don't clobber each other.
  const storageKey = isReview ? "bw-review-filter" : "bw-feed-filter";
  const defaultFilter: Filter = isReview ? { mode: "nab" } : { mode: "diversity" };
  const allowedModes = isReview
    ? new Set(["awaiting_review", "nab", "bad_quality", "binary_nab"])
    : new Set(["diversity", "all", "interesting", "unidentified", "species"]);
  const [filter, setFilter] = useState<Filter>(() => {
    try {
      const raw = sessionStorage.getItem(storageKey);
      const parsed = raw ? (JSON.parse(raw) as Filter) : null;
      return parsed &&
        typeof parsed.mode === "string" &&
        allowedModes.has(parsed.mode)
        ? parsed
        : defaultFilter;
    } catch {
      return defaultFilter;
    }
  });
  useEffect(() => {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(filter));
    } catch {
      /* private mode — ignore */
    }
  }, [storageKey, filter]);

  // "Rollups" / "Encounters" — feed surface only. Defaults to true so
  // chronological browsing groups bursts into encounters out of the box.
  const [rollups, setRollups] = useState<boolean>(() => {
    try {
      const val = sessionStorage.getItem("bw-feed-rollups");
      if (val !== null) return val === "1";
      return true;
    } catch {
      return true;
    }
  });
  useEffect(() => {
    try {
      sessionStorage.setItem("bw-feed-rollups", rollups ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [rollups]);

  const isDiversity = filter.mode === "diversity";

  const {
    data,
    isLoading,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    refetch,
  } = useInfiniteQuery({
    queryKey: ["detections", "feed", surface, filter],
    queryFn: ({ pageParam }) =>
      fetchDetections({
        limit: PAGE_SIZE,
        before: pageParam || undefined,
        diversity: isDiversity,
        only_not_a_bird: filter.mode === "nab",
        only_unidentified: filter.mode === "unidentified",
        interesting: filter.mode === "interesting",
        awaiting_review: filter.mode === "awaiting_review",
        species_name: filter.mode === "species" ? filter.name : undefined,
        bad_quality: filter.mode === "bad_quality",
        binary_nab: filter.mode === "binary_nab",
      }),
    initialPageParam: "" as string,
    getNextPageParam: (lastPage: Detection[]) => {
      if (lastPage.length < PAGE_SIZE) return undefined;
      return lastPage[lastPage.length - 1].cursor;
    },
  });

  // Refresh the first page periodically so new detections appear at the top.
  useEffect(() => {
    const id = setInterval(() => refetch(), 30_000);
    return () => clearInterval(id);
  }, [refetch]);

  // Infinite-scroll sentinel.
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!hasNextPage) return;
    const el = sentinelRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting && !isFetchingNextPage) {
          fetchNextPage();
        }
      },
      { rootMargin: "200px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  const rawDetections = useMemo(() => data?.pages.flat() ?? [], [data]);

  const cards = useMemo(() => {
    if (isReview) {
      return rawDetections.map((det) => ({
        det,
        count: 1,
        label: undefined,
        timeRange: undefined,
        allCrops: [det],
      }));
    }
    if (isDiversity) {
      return rawDetections.map((det) => ({
        det,
        count: det.daily_count ?? 1,
        label:
          det.daily_count && det.daily_count > 1
            ? `Best of ${det.daily_count} today`
            : undefined,
        timeRange: undefined,
        allCrops: [det],
      }));
    }
    if (rollups) {
      return collapseEncounters(rawDetections, 15);
    }
    return rawDetections.map((det) => ({
      det,
      count: 1,
      label: undefined,
      timeRange: undefined,
      allCrops: [det],
    }));
  }, [rawDetections, isReview, isDiversity, rollups]);

  if (error) return <p className="text-rust mt-4">Failed to load detections.</p>;

  const isFiltered = filter.mode !== "all" && filter.mode !== "diversity";

  // Sticky toolbar — filter (left) + Encounters + Select (right). Negative
  // margins so the blurred sticky background covers the full content width.
  const toolbar = (
    <div className="sticky top-0 z-30 -mx-4 px-4 py-2.5 mb-1 flex items-center justify-between gap-2 border-b border-line bg-[color-mix(in_oklab,var(--bg)_86%,transparent)] backdrop-blur">
      <FilterPicker
        value={filter}
        onChange={setFilter}
        surface={surface}
      />
      <div className="flex items-center gap-2">
        {!isReview && !isDiversity && (
          <button
            className={`rounded-full px-3.5 py-1.5 text-[12.5px] font-semibold border transition-colors ${
              rollups
                ? "text-surface border-leaf"
                : "bg-surface text-muted border-line hover:border-leaf hover:text-leaf"
            }`}
            style={rollups ? { background: "var(--accent)" } : undefined}
            onClick={() => setRollups((r) => !r)}
            aria-pressed={rollups}
            title={
              rollups
                ? "Encounters on: visits within 15m are grouped into single cards. Click to show all individual shots."
                : "Encounters off: showing every individual shot. Click to group into encounters."
            }
          >
            {rollups ? "Encounters" : "All shots"}
          </button>
        )}
        <button
          className={`rounded-full px-3.5 py-1.5 text-[12.5px] font-semibold border transition-colors ${
            batchMode
              ? "text-surface border-leaf"
              : "bg-surface text-muted border-line hover:border-leaf hover:text-leaf"
          }`}
          style={batchMode ? { background: "var(--accent)" } : undefined}
          onClick={() => (batchMode ? exitBatchMode() : setBatchMode(true))}
          aria-pressed={batchMode}
        >
          {batchMode ? "Done" : "Select"}
        </button>
      </div>
    </div>
  );

  return (
    <div>
      {!isReview && (
        <DailyStoryBulletin
          selectedSpecies={filter.mode === "species" ? filter.name : undefined}
          onSelectSpecies={(name) => {
            if (name) {
              setFilter({ mode: "species", name });
            } else {
              setFilter({ mode: "diversity" });
            }
          }}
        />
      )}
      {toolbar}
      {filter.mode === "nab" && (
        <div className="mt-3 mb-3 px-3.5 py-2.5 rounded-card border border-[color-mix(in_oklab,var(--rust)_35%,var(--line))] bg-[color-mix(in_oklab,var(--rust)_8%,var(--card))] text-sm text-ink">
          <strong className="font-semibold">Reviewing past 'Not a bird' labels.</strong>{" "}
          Use 'Wrong species?' on any crop to re-correct it — it'll move back into the
          main feed (or get re-labeled). The active-learning training set updates immediately.
        </div>
      )}

      {isLoading ? (
        <p className="text-muted mt-4">Loading…</p>
      ) : cards.length === 0 ? (
        <div className="text-center py-14">
          <p className="font-serif italic text-xl text-muted">
            {filter.mode === "species"
              ? `No matches for "${filter.name}".`
              : filter.mode === "nab"
                ? "No NAB labels to review."
                : isFiltered
                  ? "No matches for this filter."
                  : "No birds yet today."}
          </p>
          <p className="text-sm text-faint mt-1.5">
            {filter.mode === "nab"
              ? "If you mark a detection as 'Not a bird' in the feed, it will appear here for review."
              : isFiltered
                ? "Try changing the filter at the top."
                : "Once the camera fires a motion event, detections will appear here."}
          </p>
        </div>
      ) : (
        <>
          {/* Multi-column grid: shrinking each crop smooths over the feeder-cam's
              motion blur / low resolution. */}
          <div className="mt-3 grid gap-3.5 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {cards.map(({ det, count, label, timeRange, allCrops }) => (
              <DetectionCard
                key={det.id}
                detection={det}
                selected={batchMode ? selectedIds.has(det.id) : undefined}
                onToggleSelect={batchMode ? () => toggleSelect(det.id) : undefined}
                reviewMode={filter.mode === "awaiting_review"}
                seriesCount={count}
                seriesLabel={label}
                timeRange={timeRange}
                encounterCrops={allCrops}
              />
            ))}
          </div>
          {batchMode && (
            <BulkActionBar selectedIds={[...selectedIds]} onClear={exitBatchMode} />
          )}

          <div
            ref={sentinelRef}
            className="py-7 text-center text-xs tracking-wide text-faint"
          >
            {isFetchingNextPage
              ? "Loading more…"
              : hasNextPage
                ? "Scroll for more"
                : `— end of feed · ${cards.length} ${
                    isDiversity ? "species" : rollups ? "encounter" : "detection"
                  }${cards.length === 1 ? "" : "s"} —`}
          </div>
        </>
      )}
    </div>
  );
}
