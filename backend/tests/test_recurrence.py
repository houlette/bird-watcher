"""Unit tests for the recurring-box fixture detector."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from pipeline import recurrence


@dataclass
class _FakeDet:
    bbox: tuple[int, int, int, int]
    confidence: float


def _rows(box, days, per_day, start_id=0):
    """`per_day` detections on the same box for each of `days` days."""
    out = []
    i = start_id
    for d in range(days):
        for _ in range(per_day):
            out.append((i, box, date(2026, 5, 1 + d)))
            i += 1
    return out


def test_iou_is_one_for_identical_boxes():
    assert recurrence.iou((10, 10, 50, 50), (10, 10, 50, 50)) == pytest.approx(1.0)


def test_iou_is_zero_for_disjoint_boxes():
    assert recurrence.iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_recurring_box_across_enough_days_qualifies():
    rows = _rows((1000, 1000, 80, 80), days=recurrence.MIN_DISTINCT_DAYS,
                 per_day=recurrence.MIN_CLUSTER_SIZE)
    fixtures = recurrence.qualifying(recurrence.cluster(rows))
    assert len(fixtures) == 1
    assert fixtures[0].n == recurrence.MIN_DISTINCT_DAYS * recurrence.MIN_CLUSTER_SIZE


def test_one_busy_day_is_not_a_fixture():
    """The failure mode this guards: a bird that stays put for one long
    session looks exactly like scenery until you ask how many days."""
    rows = _rows((1000, 1000, 80, 80), days=1, per_day=200)
    assert recurrence.qualifying(recurrence.cluster(rows)) == []


def test_a_few_visits_across_many_days_is_not_a_fixture():
    """A real bird reusing a favourite perch. Spread over days, but nowhere
    near the volume a fixed object racks up."""
    rows = _rows((1000, 1000, 80, 80), days=10, per_day=1)
    assert recurrence.qualifying(recurrence.cluster(rows)) == []


def test_drifting_boxes_do_not_merge_into_one_fixture():
    """Overlap has to be tight. Two perches 60px apart are different things
    even though a looser threshold would join them."""
    a = _rows((1000, 1000, 80, 80), days=3, per_day=20)
    b = _rows((1060, 1000, 80, 80), days=3, per_day=20, start_id=10_000)
    fixtures = recurrence.qualifying(recurrence.cluster(a + b))
    assert len(fixtures) == 2


def test_bucketing_does_not_lose_matches_across_a_bucket_edge():
    """Boxes straddling a bucket boundary must still cluster, otherwise the
    optimisation silently changes the answer."""
    edge = recurrence.BUCKET_PX - 2
    rows = _rows((edge, edge, 8, 8), days=3, per_day=20)
    assert len(recurrence.qualifying(recurrence.cluster(rows))) == 1


def test_filter_suppresses_a_detection_on_a_fixture_box():
    boxes = [(1000, 1000, 80, 80)]
    d = _FakeDet(bbox=(1000, 1000, 80, 80), confidence=0.40)
    kept, suppressed = recurrence.filter_detections([d], boxes)
    assert kept == [] and suppressed == 1


def test_filter_respects_the_confidence_override():
    boxes = [(1000, 1000, 80, 80)]
    d = _FakeDet(bbox=(1000, 1000, 80, 80), confidence=0.90)
    kept, suppressed = recurrence.filter_detections([d], boxes)
    assert kept == [d] and suppressed == 0


def test_filter_leaves_detections_elsewhere_alone():
    boxes = [(1000, 1000, 80, 80)]
    d = _FakeDet(bbox=(2000, 300, 80, 80), confidence=0.30)
    kept, suppressed = recurrence.filter_detections([d], boxes)
    assert kept == [d] and suppressed == 0


def test_filter_with_no_fixtures_is_passthrough():
    d = _FakeDet(bbox=(1000, 1000, 80, 80), confidence=0.10)
    kept, suppressed = recurrence.filter_detections([d], [])
    assert kept == [d] and suppressed == 0
