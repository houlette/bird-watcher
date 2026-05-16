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

    species = db.query(Species).filter(Species.common_name == name).one_or_none()
    if species is None:
        species = Species(common_name=name, scientific_name="", is_rare=False)
        db.add(species)
        db.flush()

    db.add(Correction(detection_id=detection.id, correct_species_id=species.id))
    # Update the detection in-place so the feed reflects the correction immediately.
    detection.species_id = species.id
    db.commit()
    return {"ok": True, "species_id": species.id, "species": species.common_name}
