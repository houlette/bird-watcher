"""Read-side API used by the PWA."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session, joinedload

from db.models import NOT_A_BIRD_LABEL, Detection, Species, Visit
from db.session import get_db

router = APIRouter()


@router.get("")
async def list_detections(
    limit: int = Query(50, le=200),
    species_id: int | None = None,
    include_not_a_bird: bool = Query(False, description="Include detections corrected to 'Not a bird'"),
    before_id: int | None = Query(
        None,
        description="Cursor for pagination: return detections with id < before_id (older rows). "
        "Omit on the first page; pass the last detection's id from the previous page after that.",
    ),
    db: Session = Depends(get_db),
) -> list[dict]:
    q = (
        db.query(Detection)
        .options(joinedload(Detection.species), joinedload(Detection.visit))
        .order_by(desc(Detection.id))  # id is monotone and matches created_at order
    )
    if species_id is not None:
        q = q.filter(Detection.species_id == species_id)
    if not include_not_a_bird:
        # Hide detections the user has marked as not-a-bird so the feed
        # shows only real birds. The rows still exist in the DB for the
        # retraining pipeline.
        q = q.outerjoin(Species, Detection.species_id == Species.id).filter(
            (Species.common_name.is_(None)) | (Species.common_name != NOT_A_BIRD_LABEL)
        )
    if before_id is not None:
        q = q.filter(Detection.id < before_id)
    rows = q.limit(limit).all()

    return [
        {
            "id": d.id,
            "visit_id": d.visit_id,
            "species": d.species.common_name if d.species else None,
            "scientific_name": d.species.scientific_name if d.species else None,
            "confidence": d.confidence,
            "audio_confirmed": bool(d.audio_confirmed),
            "raw_predictions": d.raw_predictions,
            "crop_url": f"/media/{d.crop_path}",
            "bbox": d.bbox,
            "track_id": d.track_id,
            "created_at": d.created_at.isoformat(),
        }
        for d in rows
    ]


@router.get("/visits/{visit_id}")
async def get_visit(visit_id: int, db: Session = Depends(get_db)) -> dict:
    visit = (
        db.query(Visit)
        .options(joinedload(Visit.detections).joinedload(Detection.species))
        .filter(Visit.id == visit_id)
        .one()
    )
    return {
        "id": visit.id,
        "started_at": visit.started_at.isoformat(),
        "ended_at": visit.ended_at.isoformat() if visit.ended_at else None,
        "clip_url": f"/media/{visit.clip_path}" if visit.clip_path else None,
        "detections": [
            {
                "id": d.id,
                "species": d.species.common_name if d.species else None,
                "confidence": d.confidence,
                "audio_confirmed": bool(d.audio_confirmed),
                "crop_url": f"/media/{d.crop_path}",
                "track_id": d.track_id,
            }
            for d in visit.detections
        ],
    }
