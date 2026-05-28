import { useCallback, useEffect, useRef, useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";

import BulkActionBar from "../components/BulkActionBar";
import DetectionCard from "../components/DetectionCard";
import { fetchDetections, type Detection } from "../lib/api";

const PAGE_SIZE = 50;

type Props = {
  // "default": the normal feed (hides NAB).
  // "nab": review-mode showing ONLY 'Not a bird'-labeled crops so the user
  //   can audit past labels and re-correct mistakes via the SpeciesPicker.
  mode?: "default" | "nab";
};

export default function Feed({ mode = "default" }: Props = {}) {
  const isNabReview = mode === "nab";

  // Selection state for bulk labeling. A Set keeps add/remove and "is it in
  // here" cheap as the user scrolls through many cards. Selection persists
  // through infinite-scroll page loads and the 30s feed refetch — only an
  // explicit cancel / successful bulk submit clears it.
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const toggleSelect = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);
  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);

  const {
    data,
    isLoading,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    refetch,
  } = useInfiniteQuery({
    queryKey: ["detections", "feed", mode],
    queryFn: ({ pageParam }) =>
      fetchDetections({
        limit: PAGE_SIZE,
        before: pageParam || undefined,
        only_not_a_bird: isNabReview,
      }),
    initialPageParam: "" as string,
    getNextPageParam: (lastPage: Detection[]) => {
      // The page is empty (or smaller than PAGE_SIZE → last page reached).
      if (lastPage.length < PAGE_SIZE) return undefined;
      // Cursor for the next request is the last (oldest) row's compound cursor
      // (capture-time | detection-id). The backend uses it to fetch the next
      // chronologically-older page.
      return lastPage[lastPage.length - 1].cursor;
    },
  });

  // Refresh the first page periodically so newly-captured detections appear
  // at the top without the user reloading.
  useEffect(() => {
    const id = setInterval(() => refetch(), 30_000);
    return () => clearInterval(id);
  }, [refetch]);

  // Intersection sentinel: when the placeholder at the bottom scrolls into
  // view, fetch the next page automatically.
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
      { rootMargin: "200px" }, // start loading slightly before fully in view
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  if (isLoading) return <p className="text-slate-500">Loading…</p>;
  if (error) return <p className="text-red-600">Failed to load detections.</p>;

  const detections = data?.pages.flat() ?? [];

  if (detections.length === 0) {
    return (
      <div className="text-slate-500 text-center py-10">
        <p className="text-lg">{isNabReview ? "No NAB labels to review." : "No birds yet."}</p>
        <p className="text-sm">
          {isNabReview
            ? "If you mark a detection as 'Not a bird' in the feed, it will appear here for review."
            : "Once the camera fires a motion event, detections will appear here."}
        </p>
      </div>
    );
  }

  return (
    <div>
      {isNabReview && (
        <div className="mb-3 px-3 py-2 bg-amber-50 border border-amber-200 rounded text-sm text-amber-900">
          <strong>Reviewing past 'Not a bird' labels.</strong> Use 'Wrong species?' on any crop
          to re-correct it — it'll move back into the main feed (or get re-labeled to a real
          species). The active-learning training set updates immediately.
        </div>
      )}
      {/* Multi-column grid: shrinking each crop hides the underlying motion
          blur / low-res-ness of the feeder-cam footage — at ~180-200 px wide
          the eye smooths over artifacts that are obvious at full width. */}
      <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
        {detections.map((d) => (
          <DetectionCard
            key={d.id}
            detection={d}
            selected={selectedIds.has(d.id)}
            onToggleSelect={() => toggleSelect(d.id)}
          />
        ))}
      </div>
      <BulkActionBar selectedIds={[...selectedIds]} onClear={clearSelection} />

      <div ref={sentinelRef} className="py-6 text-center text-sm text-slate-400">
        {isFetchingNextPage
          ? "Loading more…"
          : hasNextPage
            ? "Scroll for more"
            : `End of feed (${detections.length} detection${detections.length === 1 ? "" : "s"})`}
      </div>
    </div>
  );
}
