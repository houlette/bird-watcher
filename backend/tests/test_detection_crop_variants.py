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
    res_flags = client.get(f"/api/detections/{det.id}/crop?chroma=1&clahe=0&mertens=1&sharpen=0")
    assert res_flags.status_code == 200
    mertens_custom = cv2.imdecode(np.frombuffer(res_flags.content, np.uint8), cv2.IMREAD_COLOR)
    assert mertens_custom.shape == (100, 100, 3)

    # 4b. Preset "mertens_hdr"
    res_mhdr = client.get(f"/api/detections/{det.id}/crop?preset=mertens_hdr")
    assert res_mhdr.status_code == 200
    mhdr = cv2.imdecode(np.frombuffer(res_mhdr.content, np.uint8), cv2.IMREAD_COLOR)
    assert mhdr.shape == (100, 100, 3)
    assert not np.array_equal(decoded_raw, mhdr)

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


def test_crop_variants_super_res(client, db, monkeypatch, tmp_path):
    """Test getting super_res preset and verify resolution scaling."""
    import pipeline.process as proc

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir(parents=True)

    monkeypatch.setattr(proc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(proc, "CROPS_DIR", crops_dir)

    v = Visit(clip_path="clips/dummy3.mp4")
    db.add(v)
    db.flush()

    # 60x80 synthetic crop
    raw_img = np.full((60, 80, 3), 110, dtype=np.uint8)
    raw_path = crops_dir / f"v{v.id:08d}_t0003_raw.jpg"
    cv2.imwrite(str(raw_path), raw_img)

    det = Detection(
        visit_id=v.id,
        track_id=3,
        confidence=0.95,
        crop_path=f"crops/v{v.id:08d}_t0003.jpg",
        bbox=[50, 50, 80, 60],
    )
    db.add(det)
    db.commit()

    # 1. Check variants metadata includes super_res
    res_meta = client.get(f"/api/detections/{det.id}/crop-variants")
    assert res_meta.status_code == 200
    meta = res_meta.json()
    assert "super_res" in meta["variants"]
    assert "super_res_only" in meta["variants"]
    assert meta["has_sr"] is False

    # 2. Query super_res preset (on-demand 2x scale)
    res_sr = client.get(f"/api/detections/{det.id}/crop?preset=super_res")
    assert res_sr.status_code == 200
    assert res_sr.headers["content-type"] == "image/jpeg"

    # Decode returned JPEG to verify 2x dimensions
    arr = np.frombuffer(res_sr.content, np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape == (120, 160, 3)

    # 3. Query with pre-saved _sr.jpg
    sr_file = crops_dir / f"v{v.id:08d}_t0003_sr.jpg"
    sr_img = np.full((120, 160, 3), 140, dtype=np.uint8)
    cv2.imwrite(str(sr_file), sr_img)

    res_meta2 = client.get(f"/api/detections/{det.id}/crop-variants")
    assert res_meta2.json()["has_sr"] is True

    res_sr2 = client.get(f"/api/detections/{det.id}/crop?preset=super_res_only")
    assert res_sr2.status_code == 200
    arr2 = np.frombuffer(res_sr2.content, np.uint8)
    decoded2 = cv2.imdecode(arr2, cv2.IMREAD_COLOR)
    assert decoded2.shape == (120, 160, 3)


def test_detection_has_lucky_and_sr_flags(client, db, monkeypatch, tmp_path):
    """Verify that GET /api/detections and GET /api/visits/{id} surface has_lucky and has_sr."""
    import pipeline.process as proc
    import routers.detections as det_router

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir(parents=True)

    monkeypatch.setattr(proc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(proc, "CROPS_DIR", crops_dir)
    monkeypatch.setattr(det_router, "CROPS_DIR", crops_dir)

    v = Visit(clip_path="clips/v00000004.mp4")
    db.add(v)
    db.flush()

    d1 = Detection(
        visit_id=v.id,
        track_id=1,
        confidence=0.9,
        crop_path=f"crops/v{v.id:08d}_t0001.jpg",
        bbox=[0, 0, 100, 100],
    )
    d2 = Detection(
        visit_id=v.id,
        track_id=2,
        confidence=0.85,
        crop_path=f"crops/v{v.id:08d}_t0002.jpg",
        bbox=[0, 0, 100, 100],
    )
    db.add_all([d1, d2])
    db.commit()

    # D1 has lucky initial_raw and sr; D2 has sisr
    (crops_dir / f"v{v.id:08d}_t0001_initial_raw.jpg").write_bytes(b"dummy")
    (crops_dir / f"v{v.id:08d}_t0001_sr.jpg").write_bytes(b"dummy")
    (crops_dir / f"v{v.id:08d}_t0002_sisr.jpg").write_bytes(b"dummy")

    # 1. Test GET /api/detections
    res = client.get("/api/detections")
    assert res.status_code == 200
    items = {item["id"]: item for item in res.json()}
    assert items[d1.id]["has_lucky"] is True
    assert items[d1.id]["has_sr"] is True
    assert items[d1.id]["has_sisr"] is False
    assert items[d2.id]["has_lucky"] is False
    assert items[d2.id]["has_sr"] is False
    assert items[d2.id]["has_sisr"] is True

    # 2. Test GET /api/visits/{id}
    res_visit = client.get(f"/api/detections/visits/{v.id}")
    assert res_visit.status_code == 200
    visit_dets = {item["id"]: item for item in res_visit.json()["detections"]}
    assert visit_dets[d1.id]["has_lucky"] is True
    assert visit_dets[d1.id]["has_sr"] is True
    assert visit_dets[d1.id]["has_sisr"] is False
    assert visit_dets[d2.id]["has_lucky"] is False
    assert visit_dets[d2.id]["has_sr"] is False
    assert visit_dets[d2.id]["has_sisr"] is True


def test_crop_variants_single_image_super_res(client, db, monkeypatch, tmp_path):
    """Test single-image neural super-resolution (FSRCNN) endpoint, scaling, and presets."""
    import pipeline.process as proc

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir(parents=True)

    monkeypatch.setattr(proc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(proc, "CROPS_DIR", crops_dir)

    v = Visit(clip_path="clips/dummy5.mp4")
    db.add(v)
    db.flush()

    # 40x50 synthetic crop
    raw_img = np.full((40, 50, 3), 120, dtype=np.uint8)
    raw_path = crops_dir / f"v{v.id:08d}_t0005_raw.jpg"
    cv2.imwrite(str(raw_path), raw_img)

    det = Detection(
        visit_id=v.id,
        track_id=5,
        confidence=0.96,
        crop_path=f"crops/v{v.id:08d}_t0005.jpg",
        bbox=[10, 10, 50, 40],
    )
    db.add(det)
    db.commit()

    # 1. Check variants metadata includes neural_sr
    res_meta = client.get(f"/api/detections/{det.id}/crop-variants")
    assert res_meta.status_code == 200
    meta = res_meta.json()
    assert "neural_sr" in meta["variants"]
    assert "neural_sr_only" in meta["variants"]
    assert meta["has_sisr"] is False

    # 2. Query neural_sr preset (on-demand 2x scale)
    res_sisr = client.get(f"/api/detections/{det.id}/crop?preset=neural_sr")
    assert res_sisr.status_code == 200
    assert res_sisr.headers["content-type"] == "image/jpeg"

    # Decode returned JPEG to verify 2x dimensions (40x50 -> 80x100)
    arr = np.frombuffer(res_sisr.content, np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape == (80, 100, 3)

    # 3. Query neural_sr_only
    res_sisr_only = client.get(f"/api/detections/{det.id}/crop?preset=neural_sr_only")
    assert res_sisr_only.status_code == 200
    decoded_only = cv2.imdecode(np.frombuffer(res_sisr_only.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_only.shape == (80, 100, 3)

    # 4. Query with pre-saved _sisr.jpg
    sisr_file = crops_dir / f"v{v.id:08d}_t0005_sisr.jpg"
    sisr_img = np.full((80, 100, 3), 160, dtype=np.uint8)
    cv2.imwrite(str(sisr_file), sisr_img)

    res_meta2 = client.get(f"/api/detections/{det.id}/crop-variants")
    assert res_meta2.json()["has_sisr"] is True

    res_presaved = client.get(f"/api/detections/{det.id}/crop?preset=neural_sr_only")
    assert res_presaved.status_code == 200
    decoded_presaved = cv2.imdecode(np.frombuffer(res_presaved.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_presaved.shape == (80, 100, 3)


def test_apply_synthetic_bokeh_direct():
    """Verify _apply_synthetic_bokeh and render_crop_variant(..., bokeh=True) directly."""
    from pipeline.process import _apply_synthetic_bokeh, render_crop_variant

    # 1. Gracefully handles empty or tiny inputs
    assert _apply_synthetic_bokeh(None) is None
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert _apply_synthetic_bokeh(empty).size == 0

    tiny = np.ones((16, 16, 3), dtype=np.uint8) * 100
    out_tiny = _apply_synthetic_bokeh(tiny)
    assert np.array_equal(out_tiny, tiny)

    # 2. Renders on standard crop and produces valid image with bokeh background defocus
    img = np.full((120, 100, 3), 40, dtype=np.uint8)
    # High-contrast background noise / edges on outer perimeter
    img[:20, :] = 220
    img[-20:, :] = 220
    img[:, :15] = 220
    img[:, -15:] = 220
    # Central bird-like subject
    img[40:80, 30:70] = [35, 120, 210]

    out_bokeh = _apply_synthetic_bokeh(img, strength=1.0)
    assert out_bokeh.shape == img.shape
    assert out_bokeh.dtype == np.uint8

    # Background sharp boundary should be softened/blurred by bokeh
    assert not np.array_equal(out_bokeh, img)

    # 3. render_crop_variant with bokeh=True
    rendered = render_crop_variant(img, bokeh=True)
    assert rendered.shape == img.shape
    assert rendered.dtype == np.uint8


def test_crop_variants_synthetic_bokeh(client, db, monkeypatch, tmp_path):
    """Test synthetic bokeh crop variant presets, query params, and metadata."""
    import pipeline.process as proc

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir(parents=True)

    monkeypatch.setattr(proc, "DATA_DIR", tmp_path)
    monkeypatch.setattr(proc, "CROPS_DIR", crops_dir)

    v = Visit(clip_path="clips/dummy6.mp4")
    db.add(v)
    db.flush()

    raw_img = np.full((100, 100, 3), 90, dtype=np.uint8)
    raw_img[30:70, 30:70] = [20, 160, 230]
    raw_path = crops_dir / f"v{v.id:08d}_t0006_raw.jpg"
    cv2.imwrite(str(raw_path), raw_img)

    det = Detection(
        visit_id=v.id,
        track_id=6,
        confidence=0.94,
        crop_path=f"crops/v{v.id:08d}_t0006.jpg",
        bbox=[20, 20, 80, 80],
    )
    db.add(det)
    db.commit()

    # 1. Verify variants metadata contains bokeh presets
    res_meta = client.get(f"/api/detections/{det.id}/crop-variants")
    assert res_meta.status_code == 200
    meta = res_meta.json()
    assert "bokeh" in meta["variants"]
    assert "bokeh_only" in meta["variants"]

    # 2. Query bokeh preset
    res_bokeh = client.get(f"/api/detections/{det.id}/crop?preset=bokeh")
    assert res_bokeh.status_code == 200
    assert res_bokeh.headers["content-type"] == "image/jpeg"
    decoded_bokeh = cv2.imdecode(np.frombuffer(res_bokeh.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_bokeh.shape == (100, 100, 3)

    # 3. Query bokeh_only preset
    res_only = client.get(f"/api/detections/{det.id}/crop?preset=bokeh_only")
    assert res_only.status_code == 200
    decoded_only = cv2.imdecode(np.frombuffer(res_only.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_only.shape == (100, 100, 3)

    # 4. Query bokeh query param
    res_param = client.get(f"/api/detections/{det.id}/crop?bokeh=1&chroma=1&sharpen=1")
    assert res_param.status_code == 200
    decoded_param = cv2.imdecode(np.frombuffer(res_param.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_param.shape == (100, 100, 3)




