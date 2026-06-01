"""Populate Species.reference_image_url with Wikipedia thumbnails.

The review-mode cards in the PWA show a small reference photo next to
the species name so the user can compare their crop against a clean
shot. Wikipedia's REST API exposes a `page/summary/{title}` endpoint
that returns a `thumbnail.source` URL for most articles — usable as a
direct <img src=…> in the frontend.

Idempotent. Skips species that already have a URL or were checked
recently (within RECHECK_DAYS). Records the attempt timestamp even on
miss so we don't hammer Wikipedia for species that genuinely don't
have a page (rare or recently-split taxa).

Usage:
    cd backend
    python scripts/fetch_reference_images.py             # all missing
    python scripts/fetch_reference_images.py --limit 20  # first N
    python scripts/fetch_reference_images.py --force     # ignore cache
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.families import FAMILY_NAMES  # noqa: E402
from db.models import SENTINEL_LABELS, Species  # noqa: E402
from db.session import SessionLocal, init_db  # noqa: E402
from na_birds import NA_BIRD_SPECIES  # noqa: E402
from pipeline import calibration  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_reference_images")

# Wikipedia REST summary endpoint. Returns JSON; we read `.thumbnail.source`.
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Be a polite citizen; Wikipedia asks bot traffic to identify itself.
USER_AGENT = "BirdWatcher/1.0 (https://birdwatcher.ryanhoulette.com)"
# Stay below 200 req/s. We're fetching a few hundred species at most
# (the user's labeled set); 5/s is overkill-safe and friendly.
REQUEST_DELAY_SECONDS = 0.2
# Don't re-query a known-missing species more often than this.
RECHECK_DAYS = 30


def _fetch_thumbnail(client: httpx.Client, name: str) -> str | None:
    """Return a thumbnail URL or None. Tries the species name verbatim
    first, then falls back to `{name} (bird)` for disambiguation
    collisions (e.g. "Northern Cardinal" works, but some species share
    names with non-bird subjects)."""
    candidates = [name, f"{name} (bird)"]
    for candidate in candidates:
        title = candidate.replace(" ", "_")
        try:
            r = client.get(
                WIKIPEDIA_SUMMARY + title,
                timeout=10.0,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                follow_redirects=True,
            )
        except httpx.HTTPError as e:
            log.warning("HTTP error for %r: %s", candidate, e)
            continue
        if r.status_code == 404:
            continue   # try next candidate
        if r.status_code != 200:
            log.warning("Unexpected %d for %r", r.status_code, candidate)
            continue
        try:
            data = r.json()
        except ValueError:
            continue
        # Skip disambiguation pages — they don't have a useful single
        # thumbnail. Distinguished by `type == "disambiguation"`.
        if data.get("type") == "disambiguation":
            continue
        thumbnail = data.get("thumbnail") or {}
        source = thumbnail.get("source")
        if source:
            return source
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Cap rows processed (0=all).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore RECHECK_DAYS cache; refetch even recently-checked species.")
    parser.add_argument("--dry-run", action="store_true", help="Log what would update; don't write.")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        # Ensure every picker species has a Species row before we start
        # looking up images. The picker draws from the curated NA list +
        # the yard calibration; species that haven't been corrected to
        # yet don't exist in the DB, which would leave them URL-less in
        # the picker. Seed missing rows now so backfill covers everyone.
        picker_names: set[str] = set(NA_BIRD_SPECIES)
        cal = calibration._load_calibration()  # noqa: SLF001
        if cal and isinstance(cal.get("species"), dict):
            picker_names.update(
                name for name, info in cal["species"].items()
                if isinstance(info, dict)
                and info.get("total", 0) >= calibration.MIN_DETECTIONS_FOR_ALLOWLIST
            )
        existing = {sp.common_name for sp in db.query(Species.common_name).all()}
        seeded = 0
        for name in picker_names:
            if name in existing or name in SENTINEL_LABELS or name in FAMILY_NAMES:
                continue
            db.add(Species(common_name=name, scientific_name="", is_rare=False))
            seeded += 1
        if seeded:
            db.commit()
            log.info("Seeded %d picker-list Species row(s) lacking a DB entry", seeded)

        q = db.query(Species)
        species_rows: list[Species] = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=RECHECK_DAYS)
        for sp in q.all():
            # Skip sentinels (Not a bird / Unknown bird) — no reference
            # photo would make sense.
            if sp.common_name in SENTINEL_LABELS:
                continue
            # Skip families — they're a group, no single representative
            # image. Could revisit later (use the member-species-1
            # thumbnail) if user feedback wants it.
            if sp.common_name in FAMILY_NAMES or sp.is_family:
                continue
            # Already populated?
            if sp.reference_image_url and not args.force:
                continue
            # Recently checked but no URL? Skip unless --force.
            if (
                not args.force
                and sp.reference_image_checked_at
                and sp.reference_image_checked_at > cutoff.replace(tzinfo=None)
            ):
                continue
            species_rows.append(sp)
        if args.limit:
            species_rows = species_rows[: args.limit]
        log.info("Will check %d species (--force=%s)", len(species_rows), args.force)

        hits = 0
        misses = 0
        with httpx.Client() as client:
            for i, sp in enumerate(species_rows, 1):
                url = _fetch_thumbnail(client, sp.common_name)
                if url:
                    hits += 1
                    log.info("[%d/%d] %-30s ← %s", i, len(species_rows), sp.common_name, url)
                    if not args.dry_run:
                        sp.reference_image_url = url
                else:
                    misses += 1
                    log.info("[%d/%d] %-30s   (no Wikipedia thumb)", i, len(species_rows), sp.common_name)
                if not args.dry_run:
                    sp.reference_image_checked_at = datetime.utcnow()
                # Commit periodically so a crash doesn't lose all progress.
                if not args.dry_run and i % 25 == 0:
                    db.commit()
                if i < len(species_rows):
                    time.sleep(REQUEST_DELAY_SECONDS)
        if not args.dry_run:
            db.commit()
        log.info("Done. hits=%d misses=%d", hits, misses)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
