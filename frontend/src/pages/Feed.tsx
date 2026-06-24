import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";

import BulkActionBar from "../components/BulkActionBar";
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

// "Best only" collapse: keep one card per (visit, species). A motion burst
// often yields a dozen near-identical crops of the same bird; this folds
// them to the single best one. Keyed by species too so a visit that caught
// two different birds doesn't silently drop one of them.
function bestScore(d: Detection): [number, number] {
  // Fused confidence first; tie-break on sharpness×area (a crisp, large
  // crop beats a blurry speck at the same confidence).
  return [d.confidence, (d.sharpness ?? 0) * (d.crop_area_px ?? 0)];
}

function collapseBestPerVisit(
  dets: Detection[],
): { det: Detection; count: number }[] {
  const groups = new Map<string, { best: Detection; count: number }>();
  const order: string[] = [];
  for (const d of dets) {
    const key = `${d.visit_id}:${d.species_id ?? "u"}`;
    const g = groups.get(key);
    if (!g) {
      groups.set(key, { best: d, count: 1 });
      order.push(key);
    } else {
      g.count += 1;
      const [c1, q1] = bestScore(d);
      const [c0, q0] = bestScore(g.best);
      if (c1 > c0 || (c1 === c0 && q1 > q0)) g.best = d;
    }
  }
  return order.map((k) => {
    const g = groups.get(k)!;
    return { det: g.best, count: g.count };
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
  // don't clobber each other. We also validate the stored mode against the
  // surface's own filter set — a filter from the wrong surface (or a stale
  // value left by an older build) falls back to the default rather than
  // showing an option the picker can't even offer.
  const storageKey = isReview ? "bw-review-filter" : "bw-feed-filter";
  const defaultFilter: Filter = isReview ? { mode: "nab" } : { mode: "all" };
  const allowedModes = isReview
    ? new Set(["awaiting_review", "nab", "bad_quality", "binary_nab"])
    : new Set(["all", "unidentified", "species"]);
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

  // "Best only" — feed surface only. Persisted so it survives tab switches.
  const [bestOnly, setBestOnly] = useState<boolean>(() => {
    try {
      return sessionStorage.getItem("bw-feed-bestonly") === "1";
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      sessionStorage.setItem("bw-feed-bestonly", bestOnly ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [bestOnly]);

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
        only_not_a_bird: filter.mode === "nab",
        only_unidentified: filter.mode === "unidentified",
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
  // Best-only collapses across the whole loaded list (a visit's crops are
  // contiguous in the captured-at ordering, so even one split across a page
  // boundary regroups once both pages are in).
  const cards = useMemo(
    () =>
      bestOnly && !isReview
        ? collapseBestPerVisit(rawDetections)
        : rawDetections.map((det) => ({ det, count: 1 })),
    [rawDetections, bestOnly, isReview],
  );

  if (error) return <p className="text-rust mt-4">Failed to load detections.</p>;

  const isFiltered = filter.mode !== "all";

  // Sticky toolbar — filter (left) + Best-only + Select (right). Negative
  // margins so the blurred sticky background covers the full content width.
  const toolbar = (
    <div className="sticky top-0 z-30 -mx-4 px-4 py-2.5 mb-1 flex items-center justify-between gap-2 border-b border-line bg-[color-mix(in_oklab,var(--bg)_86%,transparent)] backdrop-blur">
      <FilterPicker
        value={filter}
        onChange={setFilter}
        surface={surface}
      />
      <div className="flex items-center gap-2">
        {!isReview && (
          <button
            className={`rounded-full px-3.5 py-1.5 text-[12.5px] font-semibold border transition-colors ${
              bestOnly
                ? "text-surface border-leaf"
                : "bg-surface text-muted border-line hover:border-leaf hover:text-leaf"
            }`}
            style={bestOnly ? { background: "var(--accent)" } : undefined}
            onClick={() => setBestOnly((b) => !b)}
            aria-pressed={bestOnly}
            title="Collapse bursts to the single best crop of each bird per visit"
          >
            Best only
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
                  : "No birds yet."}
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
            {cards.map(({ det, count }) => (
              <DetectionCard
                key={det.id}
                detection={det}
                selected={batchMode ? selectedIds.has(det.id) : undefined}
                onToggleSelect={batchMode ? () => toggleSelect(det.id) : undefined}
                reviewMode={filter.mode === "awaiting_review"}
                seriesCount={count}
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
                    bestOnly ? "bird" : "detection"
                  }${cards.length === 1 ? "" : "s"} —`}
          </div>
        </>
      )}
    </div>
  );
}
