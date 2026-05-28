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


@router.post("")
async def submit_correction(req: CorrectionRequest, db: Session = Depends(get_db)) -> dict:
    detection = db.query(Detection).filter(Detection.id == req.detection_id).one_or_none()
    if detection is None:
        raise HTTPException(404, f"detection {req.detection_id} not found")

    name = req.correct_species_name.strip()
    if not name:
        raise HTTPException(400, "correct_species_name cannot be empty")

    species = _resolve_species(db, name)
    db.add(Correction(detection_id=detection.id, correct_species_id=species.id))
    # Update the detection in-place so the feed reflects the correction immediately.
    detection.species_id = species.id
    db.commit()
    return {"ok": True, "species_id": species.id, "species": species.common_name}


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
        db.add(Correction(detection_id=d.id, correct_species_id=species.id))
        d.species_id = species.id
        results.append({"id": d.id, "species_id": species.id, "species": species.common_name})
    db.commit()
    return {"ok": True, "count": len(results), "species": species.common_name, "results": results}
