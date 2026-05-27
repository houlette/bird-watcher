"""Read-side API used by the PWA."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session, joinedload

from db.models import NOT_A_BIRD_LABEL, Detection, Species, Visit
from db.session import get_db

router = APIRouter()


def _parse_cursor(cursor: str) -> tuple[datetime, int]:
    """Decode a 'captured_at_iso|detection_id' cursor string."""
    try:
        ts_str, id_str = cursor.split("|", 1)
        return datetime.fromisoformat(ts_str), int(id_str)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid cursor: {cursor}") from exc


@router.get("")
async def list_detections(
    limit: int = Query(50, le=200),
    species_id: int | None = None,
    include_not_a_bird: bool = Query(False, description="Include detections corrected to 'Not a bird'"),
    before: str | None = Query(
        None,
        description="Cursor for pagination, format: '<captured_at_iso>|<detection_id>'. "
        "Returns rows captured strictly before this point. The detection id is the "
        "tiebreaker for visits that share a started_at value.",
    ),
    db: Session = Depends(get_db),
) -> list[dict]:
    # Sort by CAPTURE time (Visit.started_at, parsed from the Reolink filename),
    # NOT by processing time (Detection.created_at / id). During a backlog drain
    # these can differ by many hours — an old visit just processed shouldn't pop
    # to the top of the feed. Detection.id desc as a tiebreaker keeps the order
    # deterministic when multiple visits share a started_at second.
    q = (
        db.query(Detection)
        .join(Visit, Detection.visit_id == Visit.id)
        .options(joinedload(Detection.species), joinedload(Detection.visit))
        .order_by(desc(Visit.started_at), desc(Detection.id))
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
    if before is not None:
        # Compound cursor: row "comes before" the cursor iff its (started_at, id)
        # is lexicographically smaller in our DESC ordering.
        cur_ts, cur_id = _parse_cursor(before)
        q = q.filter(
            or_(
                Visit.started_at < cur_ts,
                and_(Visit.started_at == cur_ts, Detection.id < cur_id),
            )
        )
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
            # captured_at: when the camera actually recorded the bird (from the
            # Reolink filename, treated as naive UTC since we run the camera in GMT+0).
            # created_at: when this Detection row was inserted (i.e. when the
            # worker finished processing the visit). Kept for debugging.
            "captured_at": d.visit.started_at.isoformat() if d.visit else d.created_at.isoformat(),
            "created_at": d.created_at.isoformat(),
            # Cursor for the next page — caller passes this back as `before`.
            "cursor": f"{(d.visit.started_at if d.visit else d.created_at).isoformat()}|{d.id}",
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
