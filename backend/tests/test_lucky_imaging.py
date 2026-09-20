"""Tests for Lucky Imaging (temporal peak sharpness hunting from source video)."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from pipeline.detect import BirdDetection
from pipeline.process import (
    _hunt_lucky_crop,
    _laplacian_variance,
    _save_source_frames,
)
from db.models import Base, Visit, Detection
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def _create_synthetic_video(path: Path, frames: list[np.ndarray], fps: float = 20.0) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _draw_bird(img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    # High-contrast feather texture inside the bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 180, 50), -1)
    for i in range(x + 5, x + w - 5, 4):
        cv2.line(img, (i, y + 5), (i, y + h - 5), (20, 20, 20), 1)
    cv2.circle(img, (x + w // 3, y + h // 3), 4, (0, 0, 255), -1)


def test_hunt_lucky_crop_finds_sharper_frame(tmp_path):
    video_path = tmp_path / "test_clip.mp4"
    w, h = 300, 300
    bx, by, bw, bh = 100, 100, 60, 60

    # Create 5 frames:
    # Frame 0: blurry
    # Frame 1: blurry
    # Frame 2: sharp (lucky frame!)
    # Frame 3: blurry
    # Frame 4: blurry
    raw_frames = []
    for _ in range(5):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 128
        _draw_bird(frame, bx, by, bw, bh)
        raw_frames.append(frame)

    frames = [
        cv2.GaussianBlur(raw_frames[0], (9, 9), 3.0),
        cv2.GaussianBlur(raw_frames[1], (7, 7), 2.5),
        raw_frames[2],  # Sharp!
        cv2.GaussianBlur(raw_frames[3], (7, 7), 2.5),
        cv2.GaussianBlur(raw_frames[4], (9, 9), 3.0),
    ]
    _create_synthetic_video(video_path, frames)

    from pipeline.process import _extract_crop_from_image

    det_blurry = BirdDetection(bbox=(bx, by, bw, bh), confidence=0.9, frame_index=0)
    det_blurry.crop = _extract_crop_from_image(det_blurry, frames[0])

    blurry_var = _laplacian_variance(det_blurry.crop)
    improved_crop, improved_bbox, found_src_idx = _hunt_lucky_crop(
        video_path,
        det_blurry,
        step=1,
        radius=2,
    )

    assert improved_crop is not None
    assert found_src_idx == 2
    sharp_var = _laplacian_variance(improved_crop)
    assert sharp_var > blurry_var * 1.5


def test_hunt_lucky_crop_rejects_unrelated_high_contrast_frame(tmp_path):
    video_path = tmp_path / "test_clip.mp4"
    w, h = 300, 300
    bx, by, bw, bh = 100, 100, 60, 60

    frame0 = np.ones((h, w, 3), dtype=np.uint8) * 128
    _draw_bird(frame0, bx, by, bw, bh)
    frame0_blur = cv2.GaussianBlur(frame0, (7, 7), 2.5)

    # Frame 1: completely different scene with huge variance (e.g. checkerboard)
    frame1_diff = np.zeros((h, w, 3), dtype=np.uint8)
    frame1_diff[::20, :] = 255
    frame1_diff[:, ::20] = 255

    _create_synthetic_video(video_path, [frame0_blur, frame1_diff])

    from pipeline.process import _extract_crop_from_image

    det = BirdDetection(bbox=(bx, by, bw, bh), confidence=0.9, frame_index=0)
    det.crop = _extract_crop_from_image(det, frame0_blur)

    improved_crop, improved_bbox, found_src_idx = _hunt_lucky_crop(
        video_path,
        det,
        step=1,
        radius=1,
    )

    # Frame 1 has higher variance but should be rejected because phase correlation/content differs
    assert improved_crop is None
    assert found_src_idx == 0


def test_hunt_lucky_crop_handles_missing_file_and_empty_crops(tmp_path):
    det = BirdDetection(bbox=(10, 10, 20, 20), confidence=0.9, frame_index=1)
    det.crop = None

    crop, bbox, idx = _hunt_lucky_crop(tmp_path / "nonexistent.mp4", det, step=3)
    assert crop is None
    assert idx == 3

    det.crop = np.zeros((0, 0, 3), dtype=np.uint8)
    crop, bbox, idx = _hunt_lucky_crop(tmp_path / "nonexistent.mp4", det, step=3)
    assert crop is None
    assert idx == 3


def test_save_source_frames_is_source_index_flag(tmp_path):
    video_path = tmp_path / "test_frames.mp4"
    frames = [np.ones((100, 100, 3), dtype=np.uint8) * (i * 20) for i in range(10)]
    _create_synthetic_video(video_path, frames)

    saved_dir = tmp_path / "frames"
    saved_dir.mkdir()

    import pipeline.process as process_mod

    orig_dir = process_mod.FRAMES_DIR
    process_mod.FRAMES_DIR = saved_dir
    try:
        # With is_source_index=True, track 1 specifies frame 4 directly
        _save_source_frames(video_path, {1: 4}, visit_id=42, is_source_index=True)
        out_file = saved_dir / "v00000042_t0001.jpg"
        assert out_file.exists()
        read_back = cv2.imread(str(out_file))
        # Frame 4 was painted with 4 * 20 = 80
        assert abs(read_back.mean() - 80) < 5
    finally:
        process_mod.FRAMES_DIR = orig_dir


def test_process_visit_lucky_imaging_integration(tmp_path, monkeypatch):
    from pipeline.track import Track

    db_url = "sqlite:///:memory:"
    engine = create_engine(db_url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    import pipeline.process as process_mod

    orig_data_dir = process_mod.DATA_DIR
    orig_crops_dir = process_mod.CROPS_DIR
    process_mod.DATA_DIR = tmp_path
    process_mod.CROPS_DIR = tmp_path / "crops"
    process_mod.CROPS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Create video clip
        video_rel = "clip.mp4"
        video_abs = tmp_path / video_rel
        w, h = 200, 200
        bx, by, bw, bh = 50, 50, 60, 60

        f0 = np.ones((h, w, 3), dtype=np.uint8) * 128
        _draw_bird(f0, bx, by, bw, bh)
        f0_blur = cv2.GaussianBlur(f0, (9, 9), 3.0)

        f1_sharp = np.ones((h, w, 3), dtype=np.uint8) * 128
        _draw_bird(f1_sharp, bx, by, bw, bh)

        # 4 frames: f0_blur, f1_sharp, f0_blur, f0_blur
        _create_synthetic_video(video_abs, [f0_blur, f1_sharp, f0_blur, f0_blur], fps=3.0)

        visit = Visit(clip_path=video_rel)
        session.add(visit)
        session.commit()
        session.refresh(visit)

        # Track with 1 blurry detection at sampled index 0
        from pipeline.process import _extract_crop_from_image

        det = BirdDetection(bbox=(bx, by, bw, bh), confidence=0.85, frame_index=0)
        det.crop = _extract_crop_from_image(det, f0_blur)
        track = Track(track_id=1, detections=[det])

        class FakeTracker:
            def update(self, *a, **k):
                pass

            def finalize(self):
                return [track]

        monkeypatch.setattr(process_mod, "detect_birds", lambda _img, _idx, **_k: [det])
        monkeypatch.setattr(process_mod, "Tracker", FakeTracker)
        monkeypatch.setattr(process_mod, "classify_bird", lambda _img: [])
        monkeypatch.setattr(process_mod, "_save_source_frames", lambda *_a, **_k: None)
        monkeypatch.setattr(process_mod, "dispatch_for_detection", lambda *_a, **_k: 0)

        # Process visit with lucky imaging enabled
        process_mod.process_visit(visit, session)

        # The detection should have been upgraded by lucky imaging to frame 1!
        db_det = session.query(Detection).filter_by(visit_id=visit.id).one()
        assert db_det.sharpness is not None
        # Blurry crop var was ~100 or less, sharp is significantly higher
        assert db_det.sharpness > _laplacian_variance(f0_blur[by : by + bh, bx : bx + bw])

        timings = visit.timings
        assert "lucky_imaging" in timings["stages_s"]
        assert timings["counts"].get("lucky_imaging_improved") == 1

    finally:
        process_mod.DATA_DIR = orig_data_dir
        process_mod.CROPS_DIR = orig_crops_dir
        session.close()
