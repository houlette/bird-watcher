"""One-time pass: ask Claude to label detections the species classifier rejected.

The local classifier (dennisjooo/Birds-Classifier-EfficientNetB2) has
near-zero top-1 accuracy on this user's data, leaving ~400 detections in
the "Unidentified" backlog. Claude with vision is substantially better
on common North American backyard species — though it'll still err on
hard cases (Cooper's vs Sharp-shinned hawk, female finches, worn
warblers). This script:

  1. Pulls every Detection with no Correction and no classifier species_id.
  2. For each, sends the crop image to Claude with a strict prompt
     constraining the answer to:
       - yard species
       - family-level catch-alls
       - "Not a bird"
       - "Unknown bird"
     plus a self-declared HIGH/MEDIUM/LOW confidence level.
  3. **Only commits HIGH-confidence answers** as Corrections with
     `source="llm-claude"`. MEDIUM goes to a JSONL review file; LOW is
     logged and skipped.

The species list is cached (1h ephemeral) so per-image cost stays low on
the second-onward request. Default model is `claude-opus-4-8` with
adaptive thinking at `effort: "medium"`.

Usage:
    # Dry-run on 20 most-recent unidentified detections
    cd backend
    ANTHROPIC_API_KEY=... python scripts/llm_classify_unidentified.py --limit 20 --dry-run

    # Commit HIGH-confidence answers on 20 detections
    ANTHROPIC_API_KEY=... python scripts/llm_classify_unidentified.py --limit 20 --auto-commit

    # Bulk run after validation
    ANTHROPIC_API_KEY=... python scripts/llm_classify_unidentified.py --auto-commit

Resumable: `source="llm-claude"` corrections are skipped on re-run by
virtue of the "no Correction" filter. Existing user corrections are also
preserved (we only touch rows the user hasn't labeled).
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/llm_classify_unidentified.py` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

from db.families import FAMILY_MEMBERS  # noqa: E402
from db.models import (  # noqa: E402
    NOT_A_BIRD_LABEL,
    SENTINEL_LABELS,
    UNKNOWN_BIRD_LABEL,
    Correction,
    Detection,
    Species,
)
from db.session import SessionLocal, init_db  # noqa: E402
from na_birds import NA_BIRD_SPECIES  # noqa: E402
from pipeline import calibration  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm_classify_unidentified")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Under data/ so the JSONL audit trail lands on the bind-mounted volume
# and survives container rebuilds. The script's own directory at
# scripts/llm_classify_results/ would get wiped on every deploy.
RESULTS_DIR = DATA_DIR / "llm_classify_results"

# claude-opus-4-8 is the most capable model — best for the long-tail
# species-disambiguation cases. ~$5/$25 per 1M in/out tokens; per-image
# cost is dominated by the image itself (~0.5-1 KB of base64). Worth
# the spend for a one-time backlog clear.
MODEL = "claude-opus-4-8"

# Adaptive thinking lets Claude reason longer on the hard cases (similar
# species pairs) and skip thinking on the obvious ones. Effort medium
# balances cost vs. quality — high adds significant token spend on every
# image without commensurate accuracy gain at this scale.
EFFORT = "medium"

# Conservative rate: 60 req/min ≈ 1/sec leaves headroom for the API's
# vision pipeline and stays well under tier limits.
REQUEST_INTERVAL_SECONDS = 1.0

# Source tags written to Correction.source so downstream code can
# distinguish where a label came from:
#   - llm-claude        HIGH-confidence Claude call, auto-committed
#   - llm-claude-medium MEDIUM-confidence call, auto-committed and
#                       awaiting user confirm/reject in the LLM-review
#                       feed filter. Distinct tag so the review UI can
#                       surface only the ones that still need a human
#                       look (HIGH is treated as already-reviewed by
#                       virtue of the 99%+ measured accuracy).
#   - llm-claude-confirmed  Promoted by the user via the Confirm button
#                       in the review feed. Tag is upgraded so the row
#                       falls out of the review queue but stays
#                       distinguishable from user-originated labels for
#                       training-data hygiene.
CORRECTION_SOURCE_HIGH = "llm-claude"
CORRECTION_SOURCE_MEDIUM = "llm-claude-medium"
# Stable list of source tags treated as "Claude said this" — useful for
# any downstream report that aggregates LLM activity. Kept here for one
# place to update if we add more tiers.
LLM_SOURCE_TAGS = (CORRECTION_SOURCE_HIGH, CORRECTION_SOURCE_MEDIUM, "llm-claude-confirmed")


def _build_species_lists() -> tuple[list[str], list[str]]:
    """Return (yard_species, broader_na). Yard list comes from yard
    calibration when present (Haikubox-heard); falls back to the curated
    NA-backyard set. Both exclude sentinels and family catch-alls."""
    cal = calibration._load_calibration()  # noqa: SLF001
    if cal and isinstance(cal.get("species"), dict):
        yard = sorted(
            (
                name for name, info in cal["species"].items()
                if isinstance(info, dict)
                and info.get("total", 0) >= calibration.MIN_DETECTIONS_FOR_ALLOWLIST
                and name not in SENTINEL_LABELS
                and name not in FAMILY_MEMBERS
            ),
            key=lambda n: -cal["species"][n].get("total", 0),
        )
    else:
        yard = []
    yard_set_lower = {n.lower() for n in yard}
    extra = sorted(
        s for s in NA_BIRD_SPECIES
        if s.lower() not in yard_set_lower
        and s not in SENTINEL_LABELS
        and s not in FAMILY_MEMBERS
    )
    return yard, extra


def _build_system_prompt(yard: list[str], extra: list[str]) -> str:
    """A single stable string. The whole prompt is cached so each
    inference only pays for the small per-image text + the image bytes."""
    families = sorted(FAMILY_MEMBERS.keys())
    lines = [
        "You are a backyard-bird identification assistant for a residential feeder camera in Boston, MA.",
        "The camera produces tight crops around YOLO-detected bird-shaped objects. Many are real birds. Some are false positives (squirrels, leaves, sun glints, hummingbird-feeder hardware).",
        "",
        "For each image, identify what's in the crop. You MUST pick exactly one label from the allowed lists below.",
        "",
        "## Yard species (most likely — recently heard by the Haikubox at this location, sorted by frequency)",
        ", ".join(yard) if yard else "(none configured)",
        "",
        "## Family-level catch-alls (use these when you can tell the broad type but can't ID the species)",
        ", ".join(families),
        "",
        "## Broader North American species (less likely but possible)",
        ", ".join(extra),
        "",
        "## Sentinels",
        f"- `{NOT_A_BIRD_LABEL}` — anything not a bird: squirrels, leaves, glints, feeder hardware",
        f"- `{UNKNOWN_BIRD_LABEL}` — clearly a bird but you genuinely can't ID it AND a family label doesn't apply (e.g., a blurry silhouette of something not in the family list)",
        "",
        "## How to choose",
        "1. If you're confident in a specific species: pick that species.",
        "2. If you're confident in the family but not the species (Sparrow vs. Junco): pick the family.",
        "3. If you can see it's a bird but family is unclear: `Unknown bird`.",
        "4. If it's not a bird at all: `Not a bird`.",
        "5. NEVER invent species not in the allowed lists. Use families instead.",
        "",
        "## Confidence levels",
        "- HIGH: Very obvious cue (distinctive plumage, posture, silhouette, color). You'd bet on it.",
        "- MEDIUM: Probable but not certain. A few species share the same look.",
        "- LOW: Guessing from limited evidence. Default to LOW when the crop is blurry, dark, or shows only a fragment.",
        "",
        "Be calibrated — Boston backyard feeder context. Rock Pigeons and Mourning Doves are very common; Sparrows are common in season; hawks are rare. Don't see exotics where none exist.",
        "",
        "Return ONLY JSON matching the schema. No prose.",
    ]
    return "\n".join(lines)


# The JSON schema we constrain the response to. Using structured outputs
# means we never have to parse free-text "I think it's a..." — we get
# a typed dict back every time.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "The exact label from the allowed lists in the system prompt.",
        },
        "confidence": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW"],
        },
        "rationale": {
            "type": "string",
            "description": "One short sentence on what cue drove the ID.",
        },
    },
    "required": ["label", "confidence", "rationale"],
    "additionalProperties": False,
}


def _load_crop_as_base64(rel_path: str) -> tuple[str, str] | None:
    """Return (media_type, base64_data) or None if the file is gone."""
    p = DATA_DIR / rel_path
    if not p.exists():
        return None
    suffix = p.suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    data = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
    return media_type, data


def _classify_one(
    client: anthropic.Anthropic,
    detection: Detection,
    system_prompt: str,
    allowed_labels: set[str],
) -> dict | None:
    """Returns {"label": ..., "confidence": ..., "rationale": ...} or None
    on a recoverable error (so the caller can move on)."""
    crop = _load_crop_as_base64(detection.crop_path)
    if crop is None:
        log.warning("Detection %d: crop file missing (%s)", detection.id, detection.crop_path)
        return None
    media_type, b64 = crop

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            thinking={"type": "adaptive"},
            output_config={
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
            # Cache the system prompt — it's identical for every detection
            # in the run. Cuts per-request input cost by ~90%.
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Identify the subject of this crop. Return JSON per the schema.",
                        },
                    ],
                }
            ],
        )
    except anthropic.APIStatusError as e:
        log.warning("Detection %d: API error %s: %s", detection.id, e.status_code, e.message)
        return None
    except anthropic.APIConnectionError as e:
        log.warning("Detection %d: connection error: %s", detection.id, e)
        return None

    # Find the first text block and parse the JSON. The structured-output
    # constraint guarantees the first text block is valid JSON.
    text_block = next((b for b in resp.content if b.type == "text"), None)
    if text_block is None:
        log.warning("Detection %d: no text block in response", detection.id)
        return None
    try:
        parsed = json.loads(text_block.text)
    except json.JSONDecodeError:
        log.warning("Detection %d: response wasn't valid JSON: %s", detection.id, text_block.text[:200])
        return None

    label = parsed.get("label")
    if label not in allowed_labels:
        log.warning("Detection %d: label %r not in allowed set — treating as LOW", detection.id, label)
        # Don't auto-commit something off-list; degrade to LOW so the
        # gate below skips it.
        parsed["confidence"] = "LOW"
        parsed["rationale"] = f"[off-list label '{label}' — degraded] " + parsed.get("rationale", "")

    # Surface cache usage so the user can see savings accruing.
    parsed["_usage"] = {
        "input_tokens": resp.usage.input_tokens,
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "output_tokens": resp.usage.output_tokens,
    }
    return parsed


def _resolve_species_for_correction(db, name: str) -> Species:
    """Mirror corrections router's resolve-or-create. Family/sentinel rows
    already exist via init_db's seeder."""
    sp = db.query(Species).filter(Species.common_name == name).one_or_none()
    if sp is None:
        sp = Species(common_name=name, scientific_name="", is_rare=False)
        db.add(sp)
        db.flush()
    return sp


def _query_unidentified(db, limit: int | None) -> list[Detection]:
    """Detections with no classifier species AND no Correction. Most-recent
    first (so a validation run uses fresh, on-disk crops)."""
    from sqlalchemy import select

    q = (
        select(Detection)
        .where(Detection.species_id.is_(None))
        .where(~Detection.id.in_(select(Correction.detection_id)))
        .order_by(Detection.id.desc())
    )
    if limit:
        q = q.limit(limit)
    return list(db.execute(q).scalars().all())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20,
                        help="Max detections to process (default 20). Set 0 for unlimited.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print what would happen — write nothing to the DB.")
    parser.add_argument("--auto-commit", action="store_true",
                        help="Write Corrections for HIGH-confidence answers. Without this flag, all answers go to the review JSONL only.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY env var not set")
        return 2

    init_db()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    review_path = RESULTS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    log.info("Review/log file: %s", review_path)

    yard, extra = _build_species_lists()
    families = sorted(FAMILY_MEMBERS.keys())
    allowed_labels = (
        set(yard) | set(extra) | set(families) | {NOT_A_BIRD_LABEL, UNKNOWN_BIRD_LABEL}
    )
    log.info(
        "Allowed labels: %d yard species, %d extra NA species, %d families, +2 sentinels",
        len(yard), len(extra), len(families),
    )
    system_prompt = _build_system_prompt(yard, extra)

    client = anthropic.Anthropic()

    db = SessionLocal()
    try:
        dets = _query_unidentified(db, args.limit if args.limit > 0 else None)
        log.info("Found %d unidentified detection(s) to classify", len(dets))

        counts = {"high": 0, "medium": 0, "low": 0, "error": 0, "committed": 0}
        with review_path.open("w") as review_f:
            for i, det in enumerate(dets, 1):
                result = _classify_one(client, det, system_prompt, allowed_labels)
                if result is None:
                    counts["error"] += 1
                    continue
                conf = result["confidence"]
                counts[conf.lower()] += 1

                record = {
                    "detection_id": det.id,
                    "visit_id": det.visit_id,
                    "track_id": det.track_id,
                    "crop_path": det.crop_path,
                    "yolo_confidence": det.yolo_confidence,
                    "result": {k: v for k, v in result.items() if k != "_usage"},
                    "usage": result["_usage"],
                }
                review_f.write(json.dumps(record) + "\n")
                review_f.flush()

                log.info(
                    "[%d/%d] det %d → %s (%s) — %s",
                    i, len(dets), det.id, result["label"], conf,
                    result["rationale"][:80],
                )

                if conf in ("HIGH", "MEDIUM") and args.auto_commit and not args.dry_run:
                    try:
                        sp = _resolve_species_for_correction(db, result["label"])
                        det.species_id = sp.id  # mirror corrections.py behavior
                        db.add(Correction(
                            detection_id=det.id,
                            correct_species_id=sp.id,
                            source=(CORRECTION_SOURCE_HIGH if conf == "HIGH" else CORRECTION_SOURCE_MEDIUM),
                            rationale=result.get("rationale"),
                        ))
                        db.commit()
                        counts["committed"] += 1
                    except Exception:
                        db.rollback()
                        log.exception("Failed to commit Correction for detection %d", det.id)

                # Pace the loop to stay under rate limits.
                if i < len(dets):
                    time.sleep(REQUEST_INTERVAL_SECONDS)

        log.info(
            "Done. HIGH=%d, MEDIUM=%d, LOW=%d, errors=%d, committed=%d",
            counts["high"], counts["medium"], counts["low"], counts["error"], counts["committed"],
        )
        log.info("Review the run at: %s", review_path)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
