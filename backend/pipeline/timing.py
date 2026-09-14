"""Wall-clock time per stage of one visit, for Visit.timings.

A fully processed daylight visit took a median of about 180 s in September
2026 while clips arrived faster than that at midday, and the only record of
where the time went was the gap between log lines: about 98% of it inside the
frame loop, which interleaves decode, tiled YOLO, three spatial filters, crop
extraction and tracking with no timer on any of them. This is the timer.

Stages are meant to be exclusive: wrap each step once, never a step inside
another timed step, so the stage times add up. Whatever they miss is reported
as `unaccounted_s` rather than silently dropped, so a large value there means
a step is missing a timer.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import TypeVar

T = TypeVar("T")

# Bump when a stage is renamed, split or merged, so queries over
# Visit.timings can tell old rows from new ones.
TIMINGS_VERSION = 1


class StageTimer:
    def __init__(self) -> None:
        self._started = perf_counter()
        self.seconds: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.extra: dict = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += perf_counter() - start

    def iterate(self, iterable: Iterable[T], name: str) -> Iterator[T]:
        """Yield from `iterable`, charging the time spent producing each item
        (and discovering the end) to `name`. For a frame generator that is
        the decode cost, including source frames it reads and throws away."""
        it = iter(iterable)
        while True:
            start = perf_counter()
            try:
                item = next(it)
            except StopIteration:
                self.seconds[name] += perf_counter() - start
                return
            self.seconds[name] += perf_counter() - start
            yield item

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] += n

    def as_dict(self) -> dict:
        total = perf_counter() - self._started
        stages = {k: round(v, 3) for k, v in sorted(self.seconds.items())}
        return {
            "version": TIMINGS_VERSION,
            "total_s": round(total, 3),
            "unaccounted_s": round(total - sum(self.seconds.values()), 3),
            "stages_s": stages,
            "counts": dict(sorted(self.counts.items())),
            **self.extra,
        }

    def summary(self) -> str:
        """One log line, largest stage first."""
        d = self.as_dict()
        stages = sorted(d["stages_s"].items(), key=lambda kv: -kv[1])
        parts = " ".join(f"{k} {v:.1f}" for k, v in stages if v >= 0.05)
        counts = ", ".join(f"{v} {k}" for k, v in d["counts"].items())
        return f"total {d['total_s']:.1f}s | {parts} | unaccounted {d['unaccounted_s']:.1f} | {counts}"
