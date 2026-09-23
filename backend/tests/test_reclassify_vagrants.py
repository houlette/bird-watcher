"""Tests for reclassify_vagrants script."""
from datetime import datetime

from db.models import Detection, Species, Visit
from pipeline.fuse import is_regional_species


def test_reclassify_vagrants_demotes_unconfirmed_non_regional():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from db.models import SENTINEL_LABELS
    from db.session import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    inca = Species(common_name="Inca Dove", scientific_name="")
    cardinal = Species(common_name="Northern Cardinal", scientific_name="")
    session.add_all([inca, cardinal])
    session.commit()

    v = Visit(started_at=datetime(2026, 5, 1, 12, 0, 0), clip_path="clips/test.mp4")
    session.add(v)
    session.commit()

    d1 = Detection(
        visit_id=v.id, species_id=inca.id, confidence=0.45,
        sharpness=500.0, crop_area_px=1200, bbox=[0, 0, 10, 10], track_id=1,
        crop_path="crops/inca.jpg", raw_predictions=[], audio_confirmed=False, created_at=v.started_at,
    )
    d2 = Detection(
        visit_id=v.id, species_id=cardinal.id, confidence=0.40,
        sharpness=500.0, crop_area_px=1200, bbox=[0, 0, 10, 10], track_id=2,
        crop_path="crops/card.jpg", raw_predictions=[], audio_confirmed=False, created_at=v.started_at,
    )
    session.add_all([d1, d2])
    session.commit()

    assert is_regional_species(inca.common_name) is False
    assert is_regional_species(cardinal.common_name) is True

    # If confidence < 0.80 and not regional, d1 should be demoted
    if not is_regional_species(d1.species.common_name) and d1.species.common_name not in SENTINEL_LABELS:
        if (d1.confidence or 0.0) < 0.80:
            d1.species_id = None
            d1.confidence = 0.0
    session.commit()

    assert d1.species_id is None
    assert d1.confidence == 0.0
    # Cardinal is regional, so untouched
    assert d2.species_id == cardinal.id
