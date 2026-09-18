"""Queries local API tunnel to find recent video clips containing verified bird visits.

Outputs a list of relative clip paths for rsync --files-from to download only
clips with actual bird activity.

Usage:
    python3 backend/scripts/detector/list_replay_clips.py --count 60 --out backend/data/eval_stage2/target_clips.txt
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


def fetch_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "BirdWatcher-Evaluator/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50, help="Target number of unique bird visit clips")
    parser.add_argument("--out", type=Path, default=Path("backend/data/eval_stage2/target_clips.txt"))
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/api")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Querying {args.api_base}/detections to gather bird visits...")
    cursor = None
    seen_visits = set()
    collected_clips = []
    species_counts = {}

    while len(collected_clips) < args.count:
        url = f"{args.api_base}/detections?limit=100"
        if cursor:
            url += f"&before={urllib.parse.quote(cursor)}"

        try:
            dets = fetch_json(url)
        except Exception as exc:
            print(f"Error fetching {url}: {exc}")
            break

        if not dets:
            print("No more detections found.")
            break

        for d in dets:
            cursor = d.get("cursor")
            sp = d.get("species")
            vid = d.get("visit_id")

            if not sp or sp == "Not a bird" or vid in seen_visits:
                continue

            seen_visits.add(vid)

            # Fetch visit details to get clip_url
            try:
                visit_data = fetch_json(f"{args.api_base}/detections/visits/{vid}")
            except Exception as exc:
                continue

            clip_url = visit_data.get("clip_url")
            if not clip_url or not clip_url.endswith(".mp4"):
                continue

            # Convert /media/clips/... -> clips/...
            rel_path = clip_url.removeprefix("/media/")
            collected_clips.append(rel_path)
            species_counts[sp] = species_counts.get(sp, 0) + 1

            if len(collected_clips) >= args.count:
                break

        print(f"Collected {len(collected_clips)}/{args.count} bird clips so far...")

    args.out.write_text("\n".join(collected_clips) + "\n")
    print(f"\nWrote {len(collected_clips)} target clip paths to {args.out}")
    print("\nSpecies breakdown across sample:")
    for sp, cnt in sorted(species_counts.items(), key=lambda x: -x[1]):
        print(f"  {sp:<25}: {cnt}")

    return 0


if __name__ == "__main__":
    import urllib.parse
    raise SystemExit(main())
