"""Gate 1.5b watch: detect seconds per tile from Visit.timings, by backend.

Usage: python detect_watch.py SINCE_UTC [UNTIL_UTC]   (ISO, e.g. 2026-09-15T16:10)
Prints per-backend median/p90 detect per tile, visit counts, processing errors,
and detections per fully processed visit over the window. Visits skipped for
falling outside daylight are not counted.

Reads the database read-only, so it can run beside the pipeline:
    docker exec birdwatcher-api python scripts/detector/detect_watch.py 2026-09-15T17:35
"""
import json
import sqlite3
import statistics
import sys

since = sys.argv[1].replace("T", " ")
# Not "9999": processed_at has NUMERIC affinity, so SQLite would turn that into
# the integer 9999, and every text timestamp sorts above any integer.
until = sys.argv[2].replace("T", " ") if len(sys.argv) > 2 else "9999-12-31 00:00"
con = sqlite3.connect("file:/app/data/birdwatcher.db?mode=ro", uri=True)
rows = con.execute(
    "SELECT id, processed_at, timings, processing_error FROM visits "
    "WHERE processed_at >= ? AND processed_at < ? ORDER BY processed_at",
    (since, until),
).fetchall()

per_tile = {}
errors = 0
visits = 0
for vid, processed_at, timings, err in rows:
    if err and err.startswith("skipped:"):
        continue
    visits += 1
    if err:
        errors += 1
        print(f"  error visit {vid} at {processed_at}: {err[:120]}")
    if not timings:
        continue
    t = json.loads(timings)
    tiles = t.get("counts", {}).get("tiles", 0)
    detect = t.get("stages_s", {}).get("detect")
    if tiles and detect is not None:
        per_tile.setdefault(t.get("detect_backend", "?"), []).append(detect / tiles)

det = con.execute(
    "SELECT COUNT(*) FROM detections d JOIN visits v ON v.id = d.visit_id "
    "WHERE v.processed_at >= ? AND v.processed_at < ? AND v.processing_error IS NULL",
    (since, until),
).fetchone()[0]

print(f"window {since} .. {until}: {len(rows)} rows, {visits} visits processed (daylight skips excluded), "
      f"{errors} with processing_error, {det} detections, {det / max(visits - errors, 1):.3f} per visit")
for backend, xs in sorted(per_tile.items()):
    xs.sort()
    p90 = xs[int(0.9 * (len(xs) - 1))]
    print(f"  {backend}: {len(xs)} visits with timings, median detect/tile {statistics.median(xs):.3f} s, p90 {p90:.3f} s")
