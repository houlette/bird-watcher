"""Unit tests for the IoU tracker. No torch / OpenCV dependency."""
from pipeline.detect import BirdDetection
from pipeline.track import MAX_MISSED_FRAMES, Tracker, iou


def det(bbox, frame_index, conf=0.9):
    return BirdDetection(bbox=bbox, confidence=conf, frame_index=frame_index)


def test_iou_overlapping_boxes():
    # Two identical boxes → IoU 1.0
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    # Half-overlap horizontally
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == 1 / 3
    # Disjoint
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_single_bird_across_frames_yields_one_track():
    tracker = Tracker()
    # Same bird drifts a few pixels each frame.
    for i, x in enumerate([100, 105, 110, 115]):
        tracker.update(i, [det((x, 100, 50, 50), frame_index=i)])
    tracks = tracker.finalize()
    assert len(tracks) == 1
    assert len(tracks[0].detections) == 4


def test_two_distinct_birds_yield_two_tracks():
    tracker = Tracker()
    # Two birds in fixed positions, far apart, present in every frame.
    for i in range(5):
        tracker.update(
            i,
            [
                det((100, 100, 40, 40), frame_index=i),
                det((400, 100, 40, 40), frame_index=i),
            ],
        )
    tracks = tracker.finalize()
    assert len(tracks) == 2
    assert all(len(t.detections) == 5 for t in tracks)


def test_track_closes_after_missing_frames():
    tracker = Tracker()
    # Bird visible for 2 frames, then gone for MAX_MISSED_FRAMES + 1 frames.
    tracker.update(0, [det((100, 100, 50, 50), frame_index=0)])
    tracker.update(1, [det((100, 100, 50, 50), frame_index=1)])
    for i in range(2, 2 + MAX_MISSED_FRAMES + 2):
        tracker.update(i, [])
    # A new bird appears in the same spot → should be a NEW track (the old one closed).
    next_frame = 2 + MAX_MISSED_FRAMES + 2
    tracker.update(next_frame, [det((100, 100, 50, 50), frame_index=next_frame)])
    tracks = tracker.finalize()
    assert len(tracks) == 2


def test_bridges_brief_gap_within_threshold():
    tracker = Tracker()
    # Bird visible, missed for just one frame (within threshold), reappears.
    tracker.update(0, [det((100, 100, 50, 50), frame_index=0)])
    tracker.update(1, [])
    tracker.update(2, [det((102, 101, 50, 50), frame_index=2)])
    tracks = tracker.finalize()
    assert len(tracks) == 1
    assert len(tracks[0].detections) == 2


def test_best_detection_picks_highest_area_x_confidence():
    """The 'best' crop is the one with the largest area × confidence score."""
    tracker = Tracker()
    # Same bird across three frames with overlapping boxes — frame 1 has the
    # biggest box and the highest confidence, so it should win.
    tracker.update(0, [det((100, 100, 50, 50), frame_index=0, conf=0.40)])
    tracker.update(1, [det((100, 100, 80, 80), frame_index=1, conf=0.95)])
    tracker.update(2, [det((105, 100, 55, 55), frame_index=2, conf=0.50)])
    tracks = tracker.finalize()
    assert len(tracks) == 1
    best = tracks[0].best_detection
    assert best.frame_index == 1
    assert best.bbox == (100, 100, 80, 80)


def test_greedy_match_picks_higher_iou_pair():
    """If detection A could match either of two tracks, it goes to the better-IoU one."""
    tracker = Tracker()
    # Seed two existing tracks at slightly different positions.
    tracker.update(0, [det((100, 100, 50, 50), 0), det((140, 100, 50, 50), 0)])
    # In the next frame, give a single detection that overlaps the second track more.
    tracker.update(1, [det((142, 100, 50, 50), 1)])
    tracks = tracker.finalize()
    # The first track gets one missed frame; the second track gets the match.
    assert len(tracks) == 2
    second_track = next(t for t in tracks if t.detections[0].bbox == (140, 100, 50, 50))
    assert len(second_track.detections) == 2
