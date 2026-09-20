"""Unit tests for the detection crop variant endpoints."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Detection, Visit
from db.session import Base, get_db
from main import app


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def client(db):
    def override():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_crop_variants_endpoints(client, db, monkeypatch, tmp_path):
    """Test getting crop variants, presets, and metadata."""
    import pipeline.process as proc

    crops_dir = tmp_path / "crops"
    frames_dir = tmp_path / "frames"
    crops_dir.mkdir(parents=True)
    frames_dir.mkdir(parents=True)

    monkeypatch.setattr(proc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(proc, "CROPS_DIR", crops_dir)
    monkeypatch.setattr(proc, "FRAMES_DIR", frames_dir)

    # Create dummy visit and detection
    v = Visit(clip_path="clips/dummy.mp4")
    db.add(v)
    db.flush()

    # Generate a synthetic colorful raw crop
    raw_img = np.zeros((100, 100, 3), dtype=np.uint8)
    raw_img[20:80, 20:80] = [30, 90, 210]  # Bird-colored patch
    raw_path = crops_dir / f"v{v.id:08d}_t0001_raw.jpg"
    cv2.imwrite(str(raw_path), raw_img)

    # Standard polished crop
    polished_path = crops_dir / f"v{v.id:08d}_t0001.jpg"
    cv2.imwrite(str(polished_path), proc._polish_for_display(raw_img))

    det = Detection(
        visit_id=v.id,
        track_id=1,
        confidence=0.92,
        crop_path=f"crops/{polished_path.name}",
        bbox=[200, 200, 100, 100],
        sharpness=180.5,
        crop_area_px=10000,
        brightness=120.0,
    )
    db.add(det)
    db.commit()

    # 1. Metadata endpoint
    res = client.get(f"/api/detections/{det.id}/crop-variants")
    assert res.status_code == 200
    meta = res.json()
    assert meta["detection_id"] == det.id
    assert meta["has_raw"] is True
    assert "polish" in meta["variants"]
    assert "raw" in meta["variants"]
    assert "chroma_only" in meta["variants"]

    # 2. Preset "raw" returns image
    res_raw = client.get(f"/api/detections/{det.id}/crop?preset=raw")
    assert res_raw.status_code == 200
    assert res_raw.headers["content-type"] == "image/jpeg"
    raw_bytes = np.frombuffer(res_raw.content, np.uint8)
    decoded_raw = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    assert decoded_raw.shape == (100, 100, 3)

    # 3. Preset "polish" returns image differing from raw
    res_polish = client.get(f"/api/detections/{det.id}/crop?preset=polish")
    assert res_polish.status_code == 200
    polish_bytes = np.frombuffer(res_polish.content, np.uint8)
    decoded_polish = cv2.imdecode(polish_bytes, cv2.IMREAD_COLOR)
    assert decoded_polish.shape == (100, 100, 3)
    # CLAHE and sharpening alter pixels compared to raw
    assert not np.array_equal(decoded_raw, decoded_polish)

    # 4. Custom boolean flags
    res_flags = client.get(f"/api/detections/{det.id}/crop?chroma=1&clahe=0&sharpen=0")
    assert res_flags.status_code == 200
    chroma_only = cv2.imdecode(np.frombuffer(res_flags.content, np.uint8), cv2.IMREAD_COLOR)
    assert chroma_only.shape == (100, 100, 3)

    # 5. Non-existent detection returns 404
    assert client.get("/api/detections/99999999/crop").status_code == 404
    assert client.get("/api/detections/99999999/crop-variants").status_code == 404


def test_crop_variants_on_demand_extraction_from_frame(client, db, monkeypatch, tmp_path):
    """When _raw.jpg is absent, extract raw crop on-demand from source frame and cache."""
    import pipeline.process as proc

    crops_dir = tmp_path / "crops"
    frames_dir = tmp_path / "frames"
    crops_dir.mkdir(parents=True)
    frames_dir.mkdir(parents=True)

    monkeypatch.setattr(proc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(proc, "CROPS_DIR", crops_dir)
    monkeypatch.setattr(proc, "FRAMES_DIR", frames_dir)

    v = Visit(clip_path="clips/dummy2.mp4")
    db.add(v)
    db.flush()

    # Write a 400x400 synthetic source frame
    frame = np.full((400, 400, 3), 50, dtype=np.uint8)
    frame[100:200, 100:200] = [20, 180, 70]
    frame_path = frames_dir / f"v{v.id:08d}_t0002.jpg"
    cv2.imwrite(str(frame_path), frame)

    det = Detection(
        visit_id=v.id,
        track_id=2,
        confidence=0.88,
        crop_path=f"crops/v{v.id:08d}_t0002.jpg",
        bbox=[100, 100, 100, 100],
    )
    db.add(det)
    db.commit()

    # Raw file does not exist yet
    raw_path = crops_dir / f"v{v.id:08d}_t0002_raw.jpg"
    assert not raw_path.exists()

    res = client.get(f"/api/detections/{det.id}/crop?preset=raw")
    assert res.status_code == 200
    # Now raw file was extracted and cached!
    assert raw_path.exists()
    cached = cv2.imread(str(raw_path))
    assert cached.shape[0] > 0 and cached.shape[1] > 0

