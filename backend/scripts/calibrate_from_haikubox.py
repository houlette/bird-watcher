"""Build yard_priors.json from the Haikubox API.

One-shot script. Run it once after the user has set HAIKUBOX_API_KEY and
HAIKUBOX_SERIAL in .env, and re-run it occasionally (every few months) to
keep the calibration current as the bird population shifts.

Usage:

    docker compose run --rm api python scripts/calibrate_from_haikubox.py
    docker compose run --rm api python scripts/calibrate_from_haikubox.py --days 180

Outputs `data/calibration/yard_priors.json`. The shape is:

    {
      "generated_at": "...",
      "data_source": {"yearly_years": [2024, 2025], "daily_days": 365},
      "species": {
        "Northern Cardinal":      {"total": 1247, "monthly_pct": {"1": 0.08, ...}},
        "White-throated Sparrow": {"total":  413, "monthly_pct": {...}}
      }
    }

Where `monthly_pct[m]` is the fraction of *this species'* detections that
fall in month `m` (over the daily window we fetched). The yearly endpoint
gives reliable totals; the daily endpoint gives the temporal pattern.

The script is defensive about API response shapes — see ingest/haikubox.py
for why (the documented Haikubox API is thin on schema details).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Make sibling modules importable when run as `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibrate")

BASE_URL = "https://api.haikubox.com"
REQUEST_TIMEOUT_SECONDS = 30
INTER_REQUEST_SLEEP_SECONDS = 0.20  # polite to the API; well under any plausible rate limit

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration" / "yard_priors.json"

# Field-name fallbacks. The Haikubox v2 yearly-count / daily-count endpoints
# return [{"bird": <name>, "count": <int>}, ...] at the top level. We keep
# the longer-form keys as fallbacks for forward compatibility.
_SPECIES_KEYS = ("bird", "cn", "common_name", "commonName", "species", "name")
_COUNT_KEYS = ("count", "n", "total", "detections")
_DATE_KEYS = ("date", "day")


def _pick(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _client() -> httpx.Client:
    headers = {
        "Authorization": f"Bearer {settings.haikubox_api_key}",
        "Accept": "application/json",
    }
    return httpx.Client(headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)


def _get_json(client: httpx.Client, path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{BASE_URL}{path}"
    r = client.get(url, params=params)
    if r.status_code == 401:
        log.error("401 Unauthorized — API key invalid or Bearer auth is wrong")
        return None
    if r.status_code == 404:
        log.error("404 Not Found at %s (params=%s) — wrong serial?", path, params)
        return None
    r.raise_for_status()
    return r.json()


def _extract_list(payload: Any, list_keys: tuple[str, ...] = ("results", "data", "counts", "items", "detections")) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in list_keys:
            inner = payload.get(key)
            if isinstance(inner, list):
                return inner
    return []


def fetch_yearly(client: httpx.Client, year: int) -> dict[str, int]:
    """Return a dict of {species_common_name: total_count} for the given year."""
    payload = _get_json(client, f"/haikubox/{settings.haikubox_serial}/yearly-count", {"year": year})
    rows = _extract_list(payload)
    totals: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        species = _pick(row, _SPECIES_KEYS)
        count = _pick(row, _COUNT_KEYS)
        if species is None or count is None:
            continue
        try:
            totals[str(species).strip()] = int(count)
        except (TypeError, ValueError):
            continue
    log.info("Year %d: %d species, top-5 by count: %s",
             year, len(totals),
             sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:5])
    return totals


def fetch_daily(client: httpx.Client, day: date) -> dict[str, int]:
    """Return {species: count} for one day."""
    payload = _get_json(client, f"/haikubox/{settings.haikubox_serial}/daily-count", {"date": day.isoformat()})
    rows = _extract_list(payload)
    out: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        species = _pick(row, _SPECIES_KEYS)
        count = _pick(row, _COUNT_KEYS)
        if species is None or count is None:
            continue
        try:
            out[str(species).strip()] = int(count)
        except (TypeError, ValueError):
            continue
    return out


def build_priors(
    yearly_totals: dict[int, dict[str, int]],
    daily_by_month: dict[str, dict[int, int]],
) -> dict[str, dict[str, Any]]:
    """Combine yearly totals with monthly per-species counts → final priors.

    `yearly_totals[year][species]` = count for that year
    `daily_by_month[species][month]` = count summed from daily data
    """
    out: dict[str, dict[str, Any]] = {}

    # All species seen anywhere.
    all_species: set[str] = set()
    for year_totals in yearly_totals.values():
        all_species.update(year_totals)
    all_species.update(daily_by_month.keys())

    for species in all_species:
        total = sum(yt.get(species, 0) for yt in yearly_totals.values())
        monthly_counts = daily_by_month.get(species, {})
        monthly_total = sum(monthly_counts.values())

        if monthly_total > 0:
            monthly_pct = {
                str(m): round(monthly_counts.get(m, 0) / monthly_total, 4)
                for m in range(1, 13)
            }
        else:
            # No daily data for this species — keep uniform so the seasonal
            # multiplier defaults to 1.0.
            monthly_pct = {str(m): round(1 / 12, 4) for m in range(1, 13)}

        out[species] = {"total": total, "monthly_pct": monthly_pct}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build yard_priors.json from Haikubox API")
    parser.add_argument("--days", type=int, default=365,
                        help="How many days of daily-count history to fetch (default 365)")
    parser.add_argument("--years", type=int, nargs="+",
                        help="Which years' yearly-counts to fetch (default: this year + previous)")
    args = parser.parse_args()

    if not settings.haikubox_api_key or not settings.haikubox_serial:
        log.error("HAIKUBOX_API_KEY and HAIKUBOX_SERIAL must be set. Aborting.")
        return 2

    today = date.today()
    years = args.years if args.years else [today.year - 1, today.year]
    log.info("Calibrating: serial=%s***, years=%s, days=%d",
             settings.haikubox_serial[:4], years, args.days)

    yearly_totals: dict[int, dict[str, int]] = {}
    # Per-species per-month accumulator from the daily-count endpoint.
    daily_by_month: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    with _client() as client:
        # 1. Yearly counts.
        for year in years:
            yearly_totals[year] = fetch_yearly(client, year)
            time.sleep(INTER_REQUEST_SLEEP_SECONDS)

        # 2. Daily counts going backward from today.
        log.info("Fetching %d daily-count days; this takes ~%.0f s",
                 args.days, args.days * INTER_REQUEST_SLEEP_SECONDS)
        for i in range(args.days):
            day = today - timedelta(days=i)
            counts = fetch_daily(client, day)
            for species, n in counts.items():
                daily_by_month[species][day.month] += n
            if i % 30 == 0 and i > 0:
                log.info("  ... %d/%d days fetched", i, args.days)
            time.sleep(INTER_REQUEST_SLEEP_SECONDS)

    priors = build_priors(yearly_totals, daily_by_month)
    if not priors:
        log.error("No species data fetched. Check API key / serial.")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": {"yearly_years": years, "daily_days": args.days},
        "species": dict(sorted(priors.items(), key=lambda kv: -kv[1]["total"])),
    }
    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    top_10 = list(output["species"].items())[:10]
    log.info("Wrote %s with %d species. Top 10 by total count:", OUTPUT_PATH, len(priors))
    for name, info in top_10:
        log.info("  %-30s total=%d", name, info["total"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
