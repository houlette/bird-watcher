import { useEffect, useRef } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";

import DetectionCard from "../components/DetectionCard";
import { fetchDetections, type Detection } from "../lib/api";

const PAGE_SIZE = 50;

export default function Feed() {
  const {
    data,
    isLoading,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    refetch,
  } = useInfiniteQuery({
    queryKey: ["detections", "feed"],
    queryFn: ({ pageParam }) =>
      fetchDetections({ limit: PAGE_SIZE, before_id: pageParam || undefined }),
    initialPageParam: 0 as number,
    getNextPageParam: (lastPage: Detection[]) => {
      // The page is empty (or smaller than PAGE_SIZE → last page reached).
      if (lastPage.length < PAGE_SIZE) return undefined;
      // Cursor for the next request is the id of the last (oldest) row.
      return lastPage[lastPage.length - 1].id;
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
        <p className="text-lg">No birds yet.</p>
        <p className="text-sm">Once the camera fires a motion event, detections will appear here.</p>
      </div>
    );
  }

  return (
    <div>
      {/* Multi-column grid: shrinking each crop hides the underlying motion
          blur / low-res-ness of the feeder-cam footage — at ~180-200 px wide
          the eye smooths over artifacts that are obvious at full width. */}
      <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
        {detections.map((d) => (
          <DetectionCard key={d.id} detection={d} />
        ))}
      </div>

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
