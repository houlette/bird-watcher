"""The held-out yardstick and the export that must respect it."""
from __future__ import annotations

import random
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from PIL import Image
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
from scripts.train import export_binary_dataset as export
from scripts.train import heldout

BEFORE = datetime(2026, 5, 20, 12)
AFTER = datetime(2026, 7, 1, 12)


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
    """Builds visits, detections and crops with as little ceremony as the
    tests can get away with."""

    def __init__(self, db, data_dir):
        self.db = db
        self.data_dir = data_dir
        self.species = {sp.common_name: sp.id for sp in db.query(Species).all()}

    def visit(self, started_at=AFTER) -> int:
        v = Visit(started_at=started_at)
        self.db.add(v)
        self.db.flush()
        return v.id

    def detection(self, visit_id, crop=True, **kw) -> int:
        det = Detection(visit_id=visit_id, confidence=0.5, bbox=[0, 0, 1, 1], track_id=1,
                        crop_path="", **kw)
        self.db.add(det)
        self.db.flush()
        det.crop_path = f"crops/det_{det.id}.jpg"
        if crop:
            path = self.data_dir / det.crop_path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64)).save(path)
        return det.id

    def correct(self, det_id, label, created_at=AFTER, source=None):
        self.db.add(Correction(detection_id=det_id, correct_species_id=self.species[label],
                               created_at=created_at, source=source))
        self.db.flush()


@pytest.fixture()
def yard(db, tmp_path):
    return Yard(db, tmp_path)


def test_cohort_rules(db, yard):
    hard = yard.detection(yard.visit())
    yard.correct(hard, "Northern Cardinal")

    corrected_before_training = yard.detection(yard.visit())
    yard.correct(corrected_before_training, "Northern Cardinal", created_at=BEFORE)

    llm_label = yard.detection(yard.visit())
    yard.correct(llm_label, "Northern Cardinal", source="llm-claude")

    unknown = yard.detection(yard.visit())
    yard.correct(unknown, UNKNOWN_BIRD_LABEL)

    # The latest correction decides: a species later rejected is junk.
    rejected = yard.detection(yard.visit())
    yard.correct(rejected, "Northern Cardinal")
    yard.correct(rejected, NOT_A_BIRD_LABEL)

    filter_fired = yard.detection(yard.visit(), nab_override_p=0.9)
    yard.correct(filter_fired, NOT_A_BIRD_LABEL)

    heard = yard.detection(yard.visit(), audio_confirmed=1)
    heard_and_corrected = yard.detection(yard.visit(), audio_confirmed=1)
    yard.correct(heard_and_corrected, NOT_A_BIRD_LABEL, source="recurrence")

    yard.detection(yard.visit(), crop=False, audio_confirmed=1)

    trained_visit = yard.visit()
    trained = yard.detection(trained_visit)
    sibling_of_trained = yard.detection(trained_visit)
    yard.correct(sibling_of_trained, NOT_A_BIRD_LABEL)
    db.commit()

    cohorts, excluded = heldout.select_cohorts(db, {trained}, yard.data_dir)

    assert cohorts == {
        heldout.HARD_BIRDS: [hard],
        heldout.EASY_BIRDS: [heard],
        heldout.JUNK: [rejected],
    }
    assert excluded[heldout.JUNK] == {"filter_fired": 1, "visit_in_deployed_training": 1}
    assert excluded[heldout.EASY_BIRDS] == {"crop_missing": 1}


def _yardstick(**cohorts) -> heldout.Yardstick:
    return heldout.Yardstick("2026-09-14T00:00:00+00:00",
                             {c: tuple(cohorts.get(c, ())) for c in heldout.COHORTS})


def test_export_withholds_yardstick_visits_and_poor_quality(db, yard):
    held_visit = yard.visit()
    held = yard.detection(held_visit)
    yard.correct(held, NOT_A_BIRD_LABEL)
    sibling = yard.detection(held_visit)
    yard.correct(sibling, NOT_A_BIRD_LABEL)

    poor = yard.detection(yard.visit())
    yard.correct(poor, POOR_QUALITY_LABEL)

    pipeline_label = yard.detection(yard.visit())
    yard.correct(pipeline_label, NOT_A_BIRD_LABEL, source="scene-mask-backfill")

    bird = yard.detection(yard.visit())
    yard.correct(bird, "Northern Cardinal", source="llm-claude")
    junk = yard.detection(yard.visit())
    yard.correct(junk, NOT_A_BIRD_LABEL)
    db.commit()

    samples, dropped = export.collect(db, _yardstick(junk=[held]), yard.data_dir)

    assert {(s.detection_id, s.cls) for s in samples} == {(bird, "bird"), (junk, "not_a_bird")}
    assert dropped["held out: yardstick row"] == 1
    assert dropped["held out: same visit as a yardstick row"] == 1
    assert dropped[f"label {POOR_QUALITY_LABEL}"] == 1
    assert dropped["source scene-mask-backfill"] == 1


def test_split_never_puts_one_day_on_both_sides(tmp_path):
    tz = ZoneInfo("America/New_York")
    samples = [
        export.Sample(i, tmp_path, "bird", visit_id=i, captured_at=datetime(2026, 7, 1 + i % 7, (i * 5) % 24, 30))
        for i in range(200)
    ]
    key = lambda s: export.group_key(s, "day", tz)  # noqa: E731
    train, val, n_groups, n_val_groups = export.split_by_group(samples, key, 0.3, random.Random(1))

    assert len(train) + len(val) == len(samples)
    assert {key(s) for s in train}.isdisjoint({key(s) for s in val})
    assert n_val_groups >= 1 and 0 < len(val) <= 0.3 * len(samples)


def test_one_large_group_cannot_swell_val(tmp_path):
    big = [export.Sample(i, tmp_path, "bird", visit_id=0, captured_at=AFTER) for i in range(900)]
    small = [export.Sample(1000 + i, tmp_path, "bird", visit_id=1 + i // 10, captured_at=AFTER)
             for i in range(100)]
    for seed in range(20):
        _, val, _, _ = export.split_by_group(big + small, lambda s: s.visit_id, 0.1, random.Random(seed))
        assert len(val) == 100


def test_bird_only_groups_cannot_crowd_junk_out_of_val(tmp_path):
    samples = [
        export.Sample(i, tmp_path, "bird" if i < 100 else "not_a_bird", visit_id=i // 10, captured_at=AFTER)
        for i in range(200)
    ]
    for seed in range(20):
        _, val, _, _ = export.split_by_group(samples, lambda s: s.visit_id, 0.1, random.Random(seed))
        assert Counter(s.cls for s in val) == {"bird": 10, "not_a_bird": 10}


def test_split_still_fills_val_when_every_group_is_too_big(tmp_path):
    samples = [export.Sample(i, tmp_path, "bird", visit_id=i % 2 + (i < 30), captured_at=AFTER)
               for i in range(100)]
    train, val, _, n_val_groups = export.split_by_group(samples, lambda s: s.visit_id, 0.1, random.Random(0))
    assert n_val_groups == 1 and train and val


def test_local_capture_day_is_not_the_utc_day(tmp_path):
    s = export.Sample(1, tmp_path, "bird", 1, datetime(2026, 7, 2, 2, 30))
    assert export.group_key(s, "day", ZoneInfo("America/New_York")) == "2026-07-01"
    assert export.group_key(s, "visit", ZoneInfo("America/New_York")) == 1


def test_cohort_rows_drop_relabelled_and_missing(db, yard):
    kept = yard.detection(yard.visit())
    yard.correct(kept, "Northern Cardinal")
    changed_mind = yard.detection(yard.visit())
    yard.correct(changed_mind, "Northern Cardinal")
    heard = yard.detection(yard.visit(), audio_confirmed=1)
    heard_then_poor = yard.detection(yard.visit(), audio_confirmed=1)
    junk_crop_gone = yard.detection(yard.visit(), crop=False)
    yard.correct(junk_crop_gone, NOT_A_BIRD_LABEL)
    db.commit()
    ys = _yardstick(hard_birds=[kept, changed_mind], easy_birds=[heard, heard_then_poor, 999_999],
                    junk=[junk_crop_gone])

    # Labels the user adds after the freeze.
    yard.correct(changed_mind, NOT_A_BIRD_LABEL)
    yard.correct(heard_then_poor, POOR_QUALITY_LABEL)
    db.commit()

    rows, skips = heldout.cohort_rows(db, ys, yard.data_dir)

    assert [i for i, _ in rows[heldout.HARD_BIRDS]] == [kept]
    assert [i for i, _ in rows[heldout.EASY_BIRDS]] == [heard]
    assert rows[heldout.JUNK] == []
    assert skips == {
        "hard_birds:relabelled": 1,
        "easy_birds:relabelled": 1,
        "easy_birds:row_deleted": 1,
        "junk:crop_missing": 1,
    }


def test_load_refuses_to_run_without_the_frozen_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        heldout.load_yardstick(tmp_path / "missing.json")
