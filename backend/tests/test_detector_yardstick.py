"""The detector yardstick: which frames count, which days are held out, and
the check a training export uses to refuse them."""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import (
    NOT_A_BIRD_LABEL,
    POOR_QUALITY_LABEL,
    UNKNOWN_BIRD_LABEL,
    Correction,
    Detection,
    Species,
    Visit,
)
from db.session import Base
from scripts.detector import yardstick as ys

TZ = ZoneInfo("America/New_York")


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    for name in (NOT_A_BIRD_LABEL, UNKNOWN_BIRD_LABEL, POOR_QUALITY_LABEL, "Northern Cardinal"):
        s.add(Species(common_name=name, scientific_name=""))
    s.commit()
    try:
        yield s
    finally:
        s.close()


class Yard:
    def __init__(self, db):
        self.db = db
        self.species = {sp.common_name: sp.id for sp in db.query(Species).all()}
        self.names: set[str] = set()

    def frame(self, started_at, *labels, track_id=1, saved=True, source=None) -> int:
        v = Visit(started_at=started_at)
        self.db.add(v)
        self.db.flush()
        det = Detection(visit_id=v.id, confidence=0.5, bbox=[0, 0, 1, 1], track_id=track_id, crop_path="")
        self.db.add(det)
        self.db.flush()
        for label in labels:
            self.db.add(Correction(detection_id=det.id, correct_species_id=self.species[label], source=source))
            self.db.flush()
        if saved:
            self.names.add(ys.frame_name(v.id, track_id))
        return det.id


def test_frame_name_matches_the_pipeline():
    assert ys.frame_name(272065, 3) == "v00272065_t0003.jpg"


def test_frame_kind():
    assert ys.frame_kind("Northern Cardinal") == ys.BIRD
    assert ys.frame_kind(UNKNOWN_BIRD_LABEL) == ys.BIRD
    assert ys.frame_kind(NOT_A_BIRD_LABEL) == ys.JUNK
    assert ys.frame_kind(POOR_QUALITY_LABEL) is None
    assert ys.frame_kind(None) is None


def test_labelled_frames(db):
    yard = Yard(db)
    noon = datetime(2026, 7, 1, 16)
    bird = yard.frame(noon, "Northern Cardinal")
    unknown = yard.frame(noon, UNKNOWN_BIRD_LABEL)
    # The latest user label decides.
    rejected = yard.frame(noon, "Northern Cardinal", NOT_A_BIRD_LABEL)
    yard.frame(noon, "Northern Cardinal", source="llm-claude")
    yard.frame(noon, POOR_QUALITY_LABEL)
    yard.frame(noon, "Northern Cardinal", saved=False)
    yard.frame(noon)  # unlabelled
    # 01:30 UTC on 7 June is still 6 June at the camera.
    late = yard.frame(datetime(2026, 6, 7, 1, 30), NOT_A_BIRD_LABEL)

    frames, skipped = ys.labelled_frames(db, yard.names, TZ)

    kinds = {f.detection_id: (f.kind, f.day) for f in frames}
    assert kinds == {
        bird: (ys.BIRD, "2026-07-01"),
        unknown: (ys.BIRD, "2026-07-01"),
        rejected: (ys.JUNK, "2026-07-01"),
        late: (ys.JUNK, "2026-06-06"),
    }
    assert skipped == {"poor_quality": 1, "bird:frame_missing": 1}


def frames_on(days_kinds):
    return [
        ys.Frame(i, i, f"v{i:08d}_t0001.jpg", day, kind)
        for i, (day, kind) in enumerate(days_kinds, start=1)
    ]


def test_pick_heldout_days_draws_a_seeded_quarter_of_pre_june_days():
    pre = [f"2026-05-{d:02d}" for d in range(1, 13)]
    frames = frames_on([(d, ys.BIRD) for d in pre] + [("2026-07-01", ys.JUNK), ("2026-07-02", ys.BIRD)])
    binary_days = frozenset({"2026-07-01", "2026-08-15"})

    days, pool, drawn = ys.pick_heldout_days(frames, binary_days, share=0.25, seed=0)

    assert pool == pre
    assert len(drawn) == 3 and set(drawn) <= set(pre)
    assert days == set(drawn) | binary_days
    assert ys.pick_heldout_days(frames, binary_days, share=0.25, seed=0)[2] == drawn
    assert ys.pick_heldout_days(frames, binary_days, share=0.5, seed=0)[2] != drawn


def test_summarize_and_gate():
    frames = frames_on([
        ("2026-07-01", ys.BIRD), ("2026-07-01", ys.JUNK), ("2026-07-01", ys.JUNK),
        ("2026-07-02", ys.BIRD), ("2026-07-03", ys.JUNK),
    ])
    s = ys.summarize(frames, {"2026-07-01"})
    assert s["heldout_bird_frames"] == 1
    assert s["heldout_junk_frames"] == 2
    assert s["heldout_days_with_frames"] == 1
    assert s["training_bird_boxes"] == 1
    assert s["training_junk_frames"] == 1
    assert s["training_days_with_frames"] == 2
    gate = ys.gate_result(s)
    assert gate["heldout_bird_frames"] == {"value": 1, "minimum": 300, "pass": False}


def test_refuses_visits_on_heldout_local_days(tmp_path):
    path = tmp_path / "heldout_frames.json"
    path.write_text(json.dumps({
        "frozen_at": "2026-09-15T00:00:00+00:00",
        "camera_timezone": "America/New_York",
        "heldout_days": ["2026-06-06"],
        "frames": {"bird": {"12": "v00000001_t0001.jpg"}, "junk": {}},
    }))
    y = ys.load_yardstick(path)
    assert y.frames == {"bird": (12,), "junk": ()}
    assert y.refuses(datetime(2026, 6, 7, 1, 30))  # 6 June at the camera
    assert not y.refuses(datetime(2026, 6, 7, 16, 0))


def test_the_committed_freeze_is_the_one_gate_2_passed_on():
    """Refreezing after step 3 starts is not allowed; this fails if the file
    changes without the plan's status table changing with it."""
    y = ys.load_yardstick()
    assert y.frozen_at == "2026-09-15T18:32:33+00:00"
    assert len(y.heldout_days) == 39
    assert (len(y.frames[ys.BIRD]), len(y.frames[ys.JUNK])) == (534, 1234)
    assert not set(y.frames[ys.BIRD]) & set(y.frames[ys.JUNK])


def test_missing_yardstick_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ys.load_yardstick(tmp_path / "absent.json")
