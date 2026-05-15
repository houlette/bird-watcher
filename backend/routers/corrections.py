"""Active-learning corrections (Phase 6 wiring; endpoints ready for Phase 1)."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.models import Correction
from db.session import get_db

router = APIRouter()


class CorrectionRequest(BaseModel):
    detection_id: int
    correct_species_id: int


@router.post("")
async def submit_correction(req: CorrectionRequest, db: Session = Depends(get_db)) -> dict:
    db.add(Correction(detection_id=req.detection_id, correct_species_id=req.correct_species_id))
    db.commit()
    return {"ok": True}
