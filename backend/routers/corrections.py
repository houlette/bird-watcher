"""Active-learning corrections. The PWA POSTs here when the user fixes a
mis-ID'd detection; we log the (detection, corrected_species) pair for the
Phase 6 fine-tune script to consume later.

A correction also updates the Detection.species_id in-place so the feed
immediately shows the user's choice rather than the stale prediction.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.models import Correction, Detection, Species
from db.session import get_db

router = APIRouter()


def _resolve_species(db: Session, name: str) -> Species:
    """Get-or-create a Species row by display name. Caller has already
    stripped whitespace."""
    species = db.query(Species).filter(Species.common_name == name).one_or_none()
    if species is None:
        species = Species(common_name=name, scientific_name="", is_rare=False)
        db.add(species)
        db.flush()
    return species


class CorrectionRequest(BaseModel):
    detection_id: int
    # Picker emits the human-readable species name (matched against the
    # yard_priors allow-list); the backend handles get-or-create on Species.
    correct_species_name: str


def _retire_stale_review_queue_rows(db: Session, detection_id: int) -> None:
    """When the user submits a fresh correction for a detection that's
    sitting in an LLM-MEDIUM review queue, drop the queue marker so the
    detection actually falls out of that queue on the next refresh.

    Without this, the source=llm-claude-medium feed-filter join still
    matches the OLD correction row even after the user has overridden
    it (NAB, Poor Quality, picker, etc.), so the card stays visible in
    MEDIUM review forever. The user-just-spoke correction is the
    canonical record; the medium-suggestion correction is no longer
    informative.

    Only llm-claude-medium is retired — llm-claude (HIGH committed) and
    llm-claude-confirmed are kept because they're audit trail of LLM
    work the user already accepted.
    """
    (
        db.query(Correction)
        .filter(Correction.detection_id == detection_id)
        .filter(Correction.source == "llm-claude-medium")
        .delete(synchronize_session=False)
    )


@router.post("")
async def submit_correction(req: CorrectionRequest, db: Session = Depends(get_db)) -> dict:
    detection = db.query(Detection).filter(Detection.id == req.detection_id).one_or_none()
    if detection is None:
        raise HTTPException(404, f"detection {req.detection_id} not found")

    name = req.correct_species_name.strip()
    if not name:
        raise HTTPException(400, "correct_species_name cannot be empty")

    species = _resolve_species(db, name)
    _retire_stale_review_queue_rows(db, detection.id)
    # Snapshot the species_id the user is overwriting so the response can
    # tell the client what to restore on undo (the PWA's toast also keeps
    # its own snapshot, but echoing it keeps the two in lockstep).
    prev_species_id = detection.species_id
    correction = Correction(detection_id=detection.id, correct_species_id=species.id)
    db.add(correction)
    # Update the detection in-place so the feed reflects the correction immediately.
    detection.species_id = species.id
    db.commit()
    return {
        "ok": True,
        "species_id": species.id,
        "species": species.common_name,
        # Identifies the row to delete (and the value to restore) if the
        # user taps Undo on the confirmation toast.
        "correction_id": correction.id,
        "prev_species_id": prev_species_id,
    }


@router.post("/confirm/{detection_id}")
async def confirm_classifier_label(detection_id: int, db: Session = Depends(get_db)) -> dict:
    """Record a user confirmation of the classifier's existing label.

    Writes a Correction row with source="user-confirmed" and
    correct_species_id matching Detection.species_id. The Detection
    species value doesn't change — confirming means "the species the
    classifier picked is correct."

    Without this signal we can't measure true-positive rate: the only
    Corrections that exist today are disagreements (you only tap
    'Wrong species?' when the model is wrong), so the classifier
    accuracy metric is selection-biased to ~0%. Confirmations give us
    the other half of the picture, and let the eventual fine-tune use
    BOTH positive and negative labels.

    Rejects detections that already have any Correction (the picker
    flow already handled them) or that have no classifier species (the
    Unidentified queue handles those).
    """
    detection = db.query(Detection).filter(Detection.id == detection_id).one_or_none()
    if detection is None:
        raise HTTPException(404, f"detection {detection_id} not found")
    if detection.species_id is None:
        raise HTTPException(
            400,
            f"detection {detection_id} has no classifier species to confirm",
        )
    existing = (
        db.query(Correction).filter(Correction.detection_id == detection_id).one_or_none()
    )
    if existing is not None:
        raise HTTPException(
            409,
            f"detection {detection_id} already has a Correction (source={existing.source!r})",
        )

    species = db.get(Species, detection.species_id)
    correction = Correction(
        detection_id=detection_id,
        correct_species_id=detection.species_id,
        source="user-confirmed",
    )
    db.add(correction)
    db.commit()
    return {
        "ok": True,
        "detection_id": detection_id,
        "source": "user-confirmed",
        "species": species.common_name if species else None,
        # Confirming doesn't change species_id, so undo just removes this
        # Correction row; prev_species_id echoes the unchanged value.
        "correction_id": correction.id,
        "prev_species_id": detection.species_id,
    }


class UndoRequest(BaseModel):
    # The Correction row to remove — returned by /corrections,
    # /corrections/confirm, and bulk so the client can target exactly the
    # row its last action created.
    correction_id: int
    # The species_id the detection had *before* that action, restored
    # in-place. None is valid (it reverts to the Unidentified queue).
    restore_species_id: int | None = None


@router.post("/undo")
async def undo_correction(req: UndoRequest, db: Session = Depends(get_db)) -> dict:
    """Reverse a single just-made correction/confirmation.

    Backs the PWA's "Undo" toast: review labelling is fast and tapping the
    wrong button on a burst of near-identical crops is easy. We delete the
    Correction row the action created and restore Detection.species_id to
    the pre-action value the client snapshotted.

    Idempotent-ish: a missing correction_id 404s (already undone, or never
    existed), so a double-tap on Undo can't corrupt anything.
    """
    correction = (
        db.query(Correction).filter(Correction.id == req.correction_id).one_or_none()
    )
    if correction is None:
        raise HTTPException(404, f"correction {req.correction_id} not found")

    detection_id = correction.detection_id
    detection = db.query(Detection).filter(Detection.id == detection_id).one_or_none()
    if detection is not None:
        detection.species_id = req.restore_species_id
    db.delete(correction)
    db.commit()
    return {
        "ok": True,
        "detection_id": detection_id,
        "restored_species_id": req.restore_species_id,
    }


@router.post("/llm-confirm/{detection_id}")
async def confirm_llm_correction(detection_id: int, db: Session = Depends(get_db)) -> dict:
    """One-tap confirm of an LLM MEDIUM-confidence Correction.

    The PWA's LLM-review feed exposes a ✓ Confirm button on
    `source="llm-claude-medium"` cards. Tapping it calls here; we
    promote the source tag to `"llm-claude-confirmed"` so the row falls
    out of the MEDIUM review queue, and stamp `created_at` to now so
    the row sorts as freshly-acted-upon in any future audit.

    The Detection.species_id is left as-is — confirming means "the
    species the script picked is right," so no value change. Disagreeing
    with the LLM is done via the normal /api/corrections POST instead,
    which overwrites both the species and the Correction.
    """
    from datetime import datetime, timezone

    correction = (
        db.query(Correction)
        .filter(Correction.detection_id == detection_id)
        .filter(Correction.source == "llm-claude-medium")
        .one_or_none()
    )
    if correction is None:
        raise HTTPException(
            404,
            f"no llm-claude-medium correction for detection {detection_id}",
        )
    correction.source = "llm-claude-confirmed"
    correction.created_at = datetime.now(timezone.utc)
    db.commit()
    species = db.get(Species, correction.correct_species_id)
    return {
        "ok": True,
        "detection_id": detection_id,
        "source": correction.source,
        "species": species.common_name if species else None,
    }


class BulkCorrectionRequest(BaseModel):
    detection_ids: list[int]
    correct_species_name: str


@router.post("/bulk")
async def submit_bulk_correction(req: BulkCorrectionRequest, db: Session = Depends(get_db)) -> dict:
    """Apply the same label to many detections in one request.

    Active-learning supports this because many of the user's labeling tasks
    are inherently bulk — five crops of the same cardinal at the same
    feeder, or a dozen near-identical 'shadow on the deck' false-positives.
    Forcing a per-card SpeciesPicker round-trip for each one is the single
    biggest friction in the active-learning loop.

    Semantics:
      - Empty list → 400
      - Whitespace-only name → 400
      - Any unknown detection id → 404 (strict, since the picker only
        surfaces ids the user actually selected and a missing one points
        at a deeper problem)
      - All-or-nothing: validate everything first, then commit once
    """
    if not req.detection_ids:
        raise HTTPException(400, "detection_ids cannot be empty")
    name = req.correct_species_name.strip()
    if not name:
        raise HTTPException(400, "correct_species_name cannot be empty")

    detections = (
        db.query(Detection)
        .filter(Detection.id.in_(req.detection_ids))
        .all()
    )
    found_ids = {d.id for d in detections}
    missing = [i for i in req.detection_ids if i not in found_ids]
    if missing:
        raise HTTPException(404, f"detection ids not found: {missing[:10]}")

    species = _resolve_species(db, name)
    results = []
    for d in detections:
        _retire_stale_review_queue_rows(db, d.id)
        db.add(Correction(detection_id=d.id, correct_species_id=species.id))
        d.species_id = species.id
        results.append({"id": d.id, "species_id": species.id, "species": species.common_name})
    db.commit()
    return {"ok": True, "count": len(results), "species": species.common_name, "results": results}
