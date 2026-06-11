import { useCallback, useEffect, useRef, useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";

import BulkActionBar from "../components/BulkActionBar";
import DetectionCard from "../components/DetectionCard";
import FilterPicker, { type Filter } from "../components/FilterPicker";
import { fetchDetections, type Detection } from "../lib/api";

const PAGE_SIZE = 50;

type Props = {
  // "default": the normal feed (hides NAB).
  // "nab": review-mode showing ONLY 'Not a bird'-labeled crops.
  mode?: "default" | "nab";
};

export default function Feed({ mode = "default" }: Props = {}) {
  const isNabReview = mode === "nab";

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

  const [filter, setFilter] = useState<Filter>({ mode: "all" });
  const effectiveFilter: Filter = isNabReview ? { mode: "all" } : filter;

  const {
    data,
    isLoading,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    refetch,
  } = useInfiniteQuery({
    queryKey: ["detections", "feed", mode, effectiveFilter],
    queryFn: ({ pageParam }) =>
      fetchDetections({
        limit: PAGE_SIZE,
        before: pageParam || undefined,
        only_not_a_bird: isNabReview,
        only_unidentified: effectiveFilter.mode === "unidentified",
        awaiting_review: effectiveFilter.mode === "awaiting_review",
        species_name: effectiveFilter.mode === "species" ? effectiveFilter.name : undefined,
        source:
          effectiveFilter.mode === "llm_review"
            ? "llm-claude"
            : effectiveFilter.mode === "llm_medium_review"
              ? "llm-claude-medium"
              : undefined,
        bad_quality: effectiveFilter.mode === "bad_quality",
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

  if (error) return <p className="text-rust mt-4">Failed to load detections.</p>;

  const detections = data?.pages.flat() ?? [];
  const isFiltered = effectiveFilter.mode !== "all";

  // Sticky toolbar — filter (left) + Select (right). Negative margins so the
  // blurred sticky background covers the full content width.
  const toolbar = (
    <div className="sticky top-0 z-30 -mx-4 px-4 py-2.5 mb-1 flex items-center justify-between gap-2 border-b border-line bg-[color-mix(in_oklab,var(--bg)_86%,transparent)] backdrop-blur">
      <div>{!isNabReview && <FilterPicker value={filter} onChange={setFilter} />}</div>
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
  );

  return (
    <div>
      {toolbar}
      {isNabReview && (
        <div className="mt-3 mb-3 px-3.5 py-2.5 rounded-card border border-[color-mix(in_oklab,var(--rust)_35%,var(--line))] bg-[color-mix(in_oklab,var(--rust)_8%,var(--card))] text-sm text-ink">
          <strong className="font-semibold">Reviewing past 'Not a bird' labels.</strong>{" "}
          Use 'Wrong species?' on any crop to re-correct it — it'll move back into the
          main feed (or get re-labeled). The active-learning training set updates immediately.
        </div>
      )}

      {isLoading ? (
        <p className="text-muted mt-4">Loading…</p>
      ) : detections.length === 0 ? (
        <div className="text-center py-14">
          <p className="font-serif italic text-xl text-muted">
            {isFiltered
              ? `No matches for "${
                  effectiveFilter.mode === "species" ? effectiveFilter.name : "Unidentified"
                }".`
              : isNabReview
                ? "No NAB labels to review."
                : "No birds yet."}
          </p>
          <p className="text-sm text-faint mt-1.5">
            {isFiltered
              ? "Try changing the filter at the top."
              : isNabReview
                ? "If you mark a detection as 'Not a bird' in the feed, it will appear here for review."
                : "Once the camera fires a motion event, detections will appear here."}
          </p>
        </div>
      ) : (
        <>
          {/* Multi-column grid: shrinking each crop smooths over the feeder-cam's
              motion blur / low resolution. */}
          <div className="mt-3 grid gap-3.5 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {detections.map((d) => (
              <DetectionCard
                key={d.id}
                detection={d}
                selected={batchMode ? selectedIds.has(d.id) : undefined}
                onToggleSelect={batchMode ? () => toggleSelect(d.id) : undefined}
                reviewMode={effectiveFilter.mode === "awaiting_review"}
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
                : `— end of feed · ${detections.length} detection${
                    detections.length === 1 ? "" : "s"
                  } —`}
          </div>
        </>
      )}
    </div>
  );
}
