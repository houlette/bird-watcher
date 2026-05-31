"""Daily pipeline-funnel aggregation.

Computes a `PipelineStatsDaily` row for a given UTC date by GROUP-BYing
over Visit / Detection / Correction. The nightly job in pipeline.worker
calls `upsert_daily_stats` once per night; the /api/stats endpoint
re-runs `compute_daily_stats` for today on every request so the page
stays current.

Definitions (kept aligned with the plan; if you tweak one here, update
the docstring on PipelineStatsDaily too):

  - A "clip" is a Visit row.
  - "Daylight" excludes visits whose processing_error starts with
    "skipped:" *and* mentions "daylight". (The worker uses exactly that
    string; we match it instead of recomputing is_daylight to avoid
    drifting from the actual gate.)
  - "Detections" are all Detection rows, including species_id=NULL.
  - "Classifier-labeled" = species_id NOT NULL, AND the species isn't a
    sentinel. (When the classifier rejects, species_id is NULL; sentinels
    are only ever set via user correction, so this distinction also
    keeps NAB/Unknown out of the classifier-rate numerator.)
  - "User-corrected" = a Correction row exists for the detection.
  - Classifier accuracy = of corrections whose target species is a real
    species (not NAB / Unknown), the fraction where the Detection's
    species_id at correction time matched. Since Correction overwrites
    species_id in-place (see corrections.py), we approximate by
    comparing Detection.species_id to Correction.correct_species_id —
    they'll be equal iff the user agreed with the model. (After the
    user accepts the model's guess via the "looks right" path, the
    correction is the same species → counts as correct.)
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime, time, timedelta
from typing import Iterable

from sqlalchemy import and_, distinct, func, or_, select
from sqlalchemy.orm import Session

from db.models import (
    NOT_A_BIRD_LABEL,
    SENTINEL_LABELS,
    UNKNOWN_BIRD_LABEL,
    Correction,
    Detection,
    PipelineStatsDaily,
    Species,
    Visit,
)

log = logging.getLogger(__name__)


def _day_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime.combine(d, time.min)
    end = start + timedelta(days=1)
    return start, end


def compute_daily_stats(db: Session, d: date) -> PipelineStatsDaily:
    """Aggregate the funnel for UTC date `d` and return an unsaved
    PipelineStatsDaily instance. Caller decides whether to upsert."""
    start, end = _day_bounds(d)

    # --- Funnel: visit level -------------------------------------------------
    visits_q = db.query(Visit).filter(
        Visit.started_at >= start, Visit.started_at < end
    )
    clips_received = visits_q.count()

    # Daylight-skipped visits are flagged with "skipped: captured outside
    # daylight hours" (see worker._process_pending). Match that literal
    # prefix so a future error-message change forces us to revisit this.
    daylight_skipped = visits_q.filter(
        Visit.processing_error.like("skipped:%daylight%")
    ).count()
    clips_daylight = clips_received - daylight_skipped

    # Non-skip processing errors — operational health signal. processing_error
    # SET and processed_at SET and not a "skipped:" prefix.
    visits_with_processing_error = visits_q.filter(
        Visit.processing_error.isnot(None),
        Visit.processed_at.isnot(None),
        ~Visit.processing_error.like("skipped:%"),
    ).count()

    clips_with_detections = (
        db.query(func.count(distinct(Detection.visit_id)))
        .join(Visit, Detection.visit_id == Visit.id)
        .filter(Visit.started_at >= start, Visit.started_at < end)
        .scalar()
        or 0
    )

    # --- Funnel: detection level --------------------------------------------
    # All detection-level filters use the Visit.started_at window — i.e. a
    # detection belongs to the day its visit was captured, NOT the day the
    # worker processed it (these can differ during a backlog drain).
    base_dets = (
        db.query(Detection)
        .join(Visit, Detection.visit_id == Visit.id)
        .filter(Visit.started_at >= start, Visit.started_at < end)
    )
    detections_total = base_dets.count()

    # Classifier-labeled: the classifier produced a top-1 species at
    # processing time. We check raw_predictions[0].species rather than
    # Detection.species_id because user corrections overwrite species_id
    # in place — once the user marks a Cardinal misfire as NAB, the
    # original classifier guess is no longer visible on species_id but
    # raw_predictions still preserves it.
    #
    # JSON path expressions aren't portable across SQLite versions, so
    # we materialize the rows once and count in Python. The window is
    # already date-bounded so this is cheap.
    detections_labeled_by_classifier = 0
    raw_pred_rows = base_dets.with_entities(Detection.raw_predictions).all()
    for (raw,) in raw_pred_rows:
        if not raw:
            continue
        try:
            top = (raw[0] or {}).get("species")
        except (TypeError, AttributeError):
            top = None
        if top:
            detections_labeled_by_classifier += 1

    detections_audio_confirmed = base_dets.filter(Detection.audio_confirmed == True).count()  # noqa: E712

    # Aggregate scene-mask suppression count across all visits in window.
    # NULL on old rows (column added later) → COALESCE to 0 in SUM.
    detections_scene_mask_suppressed = int(
        visits_q.with_entities(
            func.coalesce(func.sum(Visit.scene_mask_suppressed), 0)
        ).scalar()
        or 0
    )

    # --- Correction-level metrics -------------------------------------------
    # User-corrected detections in this window.
    corrected_q = (
        db.query(Detection, Species.common_name.label("corrected_name"))
        .join(Visit, Detection.visit_id == Visit.id)
        .join(Correction, Correction.detection_id == Detection.id)
        .join(Species, Correction.correct_species_id == Species.id)
        .filter(Visit.started_at >= start, Visit.started_at < end)
    )
    corrected_rows = corrected_q.all()
    detections_user_corrected = len(corrected_rows)

    corrections_nab = sum(1 for _, n in corrected_rows if n == NOT_A_BIRD_LABEL)
    corrections_unknown = sum(1 for _, n in corrected_rows if n == UNKNOWN_BIRD_LABEL)
    corrections_real_species = detections_user_corrected - corrections_nab - corrections_unknown

    # Classifier accuracy: of corrections to a real species, how many
    # matched what the classifier had originally guessed. Correction
    # overwrites Detection.species_id, so at this point species_id ==
    # correct_species_id (i.e. "user agreed"). We need the historic
    # disagreement signal, which we recover from raw_predictions[0]:
    # that's the classifier's pre-correction top-1.
    classifier_correct = 0
    classifier_eligible = 0
    for det, corrected_name in corrected_rows:
        if corrected_name in SENTINEL_LABELS:
            continue   # NAB/Unknown — classifier had no chance
        classifier_eligible += 1
        # raw_predictions: [{"species": "...", "p": 0.55}, ...] top-1 first.
        # If raw_predictions is empty / missing, the classifier rejected the
        # crop entirely — counts as a miss (the model didn't surface this
        # species; the user had to find it).
        top1 = None
        if det.raw_predictions:
            try:
                top1 = (det.raw_predictions[0] or {}).get("species")
            except (TypeError, AttributeError):
                top1 = None
        if top1 == corrected_name:
            classifier_correct += 1

    # --- Payload: variable-shape extras -------------------------------------
    payload: dict = {}

    # Hour-of-day histogram (24 buckets) — all detections in window.
    hour_counts = [0] * 24
    for ts, n in (
        db.query(Visit.started_at, func.count(Detection.id))
        .join(Detection, Detection.visit_id == Visit.id)
        .filter(Visit.started_at >= start, Visit.started_at < end)
        .group_by(Visit.started_at)
        .all()
    ):
        if ts is not None:
            hour_counts[ts.hour] += int(n)
    payload["hour_of_day"] = hour_counts

    # YOLO-confidence histogram, split by NAB-corrected vs real-species
    # corrected. 10 buckets, 0.0–1.0 in 0.1 steps. Only includes
    # detections that have a YOLO confidence value (older rows are NULL).
    nab_hist = [0] * 10
    species_hist = [0] * 10
    for det, corrected_name in corrected_rows:
        if det.yolo_confidence is None:
            continue
        bucket = min(int(det.yolo_confidence * 10), 9)
        if corrected_name == NOT_A_BIRD_LABEL:
            nab_hist[bucket] += 1
        elif corrected_name not in SENTINEL_LABELS:
            species_hist[bucket] += 1
    payload["yolo_confidence_hist"] = {"nab": nab_hist, "species": species_hist}

    return PipelineStatsDaily(
        date=d,
        clips_received=clips_received,
        clips_daylight=clips_daylight,
        clips_with_detections=int(clips_with_detections),
        detections_total=detections_total,
        detections_labeled_by_classifier=detections_labeled_by_classifier,
        detections_user_corrected=detections_user_corrected,
        corrections_nab=corrections_nab,
        corrections_unknown=corrections_unknown,
        corrections_real_species=corrections_real_species,
        classifier_correct=classifier_correct,
        classifier_eligible=classifier_eligible,
        visits_with_processing_error=visits_with_processing_error,
        detections_audio_confirmed=detections_audio_confirmed,
        detections_scene_mask_suppressed=detections_scene_mask_suppressed,
        payload=payload,
        computed_at=datetime.utcnow(),
    )


def upsert_daily_stats(db: Session, d: date) -> PipelineStatsDaily:
    """Compute and upsert the row for date `d`. Returns the persisted row."""
    fresh = compute_daily_stats(db, d)
    existing = db.get(PipelineStatsDaily, d)
    if existing is None:
        db.add(fresh)
    else:
        for col in (
            "clips_received",
            "clips_daylight",
            "clips_with_detections",
            "detections_total",
            "detections_labeled_by_classifier",
            "detections_user_corrected",
            "corrections_nab",
            "corrections_unknown",
            "corrections_real_species",
            "classifier_correct",
            "classifier_eligible",
            "visits_with_processing_error",
            "detections_audio_confirmed",
            "detections_scene_mask_suppressed",
            "payload",
            "computed_at",
        ):
            setattr(existing, col, getattr(fresh, col))
        fresh = existing
    db.commit()
    return fresh


def compute_global_stats(db: Session) -> dict:
    """All-time totals (not bucketed by date). Used for the headline cards.

    Cheap because these are unfiltered COUNT(*) over indexed columns.
    """
    detections_total = db.query(Detection).count()
    visits_total = db.query(Visit).count()
    corrections_total = db.query(Correction).count()

    # Pending-label backlog: detections with a classifier species set and
    # no Correction. These are the queue of "model thinks this is a bird,
    # waiting on you."
    pending_backlog = (
        db.query(Detection)
        .outerjoin(Correction, Correction.detection_id == Detection.id)
        .outerjoin(Species, Detection.species_id == Species.id)
        .filter(
            Detection.species_id.isnot(None),
            Correction.id.is_(None),
            or_(
                Species.common_name.is_(None),
                ~Species.common_name.in_(SENTINEL_LABELS),
            ),
        )
        .count()
    )

    # Top species by user-corrected count + ready-to-fine-tune threshold.
    species_counts = (
        db.query(Species.common_name, func.count(Correction.id))
        .join(Correction, Correction.correct_species_id == Species.id)
        .group_by(Species.common_name)
        .order_by(func.count(Correction.id).desc())
        .all()
    )
    top_species = [
        {"species": name, "count": int(n)}
        for name, n in species_counts
        if name not in SENTINEL_LABELS
    ][:15]
    ready_to_fine_tune = sum(
        1 for name, n in species_counts if name not in SENTINEL_LABELS and n >= 50
    )

    # Per-species classifier accuracy (over all time), where we have ≥5
    # ground-truth labels. Uses the same raw_predictions[0] comparison as
    # compute_daily_stats.
    real_species_correct: Counter[str] = Counter()
    real_species_total: Counter[str] = Counter()
    for det, name in (
        db.query(Detection, Species.common_name)
        .join(Correction, Correction.detection_id == Detection.id)
        .join(Species, Correction.correct_species_id == Species.id)
        .filter(~Species.common_name.in_(SENTINEL_LABELS))
        .all()
    ):
        real_species_total[name] += 1
        top1 = None
        if det.raw_predictions:
            try:
                top1 = (det.raw_predictions[0] or {}).get("species")
            except (TypeError, AttributeError):
                top1 = None
        if top1 == name:
            real_species_correct[name] += 1

    species_accuracy = [
        {
            "species": name,
            "n": real_species_total[name],
            "accuracy": (real_species_correct[name] / real_species_total[name]),
        }
        for name in sorted(real_species_total, key=lambda k: -real_species_total[k])
        if real_species_total[name] >= 5
    ][:15]

    return {
        "visits_total": visits_total,
        "detections_total": detections_total,
        "corrections_total": corrections_total,
        "pending_backlog": pending_backlog,
        "ready_to_fine_tune_species": ready_to_fine_tune,
        "top_species": top_species,
        "species_accuracy": species_accuracy,
    }


def serialize_daily(row: PipelineStatsDaily) -> dict:
    """Convert a PipelineStatsDaily row (saved or unsaved) to the JSON
    shape the /api/stats endpoint returns."""
    classifier_accuracy = (
        row.classifier_correct / row.classifier_eligible
        if row.classifier_eligible
        else None
    )
    yolo_bird_rate = (
        row.clips_with_detections / row.clips_daylight
        if row.clips_daylight
        else None
    )
    classifier_label_rate = (
        row.detections_labeled_by_classifier / row.detections_total
        if row.detections_total
        else None
    )
    user_fp_rate = (
        row.corrections_nab / row.detections_user_corrected
        if row.detections_user_corrected
        else None
    )
    return {
        "date": row.date.isoformat(),
        "clips_received": row.clips_received,
        "clips_daylight": row.clips_daylight,
        "clips_with_detections": row.clips_with_detections,
        "detections_total": row.detections_total,
        "detections_labeled_by_classifier": row.detections_labeled_by_classifier,
        "detections_user_corrected": row.detections_user_corrected,
        "corrections_nab": row.corrections_nab,
        "corrections_unknown": row.corrections_unknown,
        "corrections_real_species": row.corrections_real_species,
        "classifier_correct": row.classifier_correct,
        "classifier_eligible": row.classifier_eligible,
        "visits_with_processing_error": row.visits_with_processing_error,
        "detections_audio_confirmed": row.detections_audio_confirmed,
        "detections_scene_mask_suppressed": row.detections_scene_mask_suppressed or 0,
        # Derived rates (caller doesn't have to recompute).
        "yolo_bird_rate": yolo_bird_rate,
        "classifier_label_rate": classifier_label_rate,
        "user_fp_rate": user_fp_rate,
        "classifier_accuracy": classifier_accuracy,
        "payload": row.payload or {},
    }


def dates_with_visits(db: Session) -> Iterable[date]:
    """All UTC dates with at least one Visit. Used by the backfill script
    and the nightly job's gap-fill pass."""
    rows = db.execute(
        select(func.date(Visit.started_at)).distinct().order_by(func.date(Visit.started_at))
    ).all()
    out: list[date] = []
    for (val,) in rows:
        if val is None:
            continue
        # SQLite returns dates as ISO strings.
        if isinstance(val, str):
            out.append(date.fromisoformat(val))
        else:
            out.append(val)
    return out
