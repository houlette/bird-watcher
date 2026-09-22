"""Security tests for static media serving: ensure database files, source code,
and unapproved directories are never exposed over HTTP.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from main import SafeStaticFiles


@pytest.fixture()
def client(tmp_path):
    safe_app = SafeStaticFiles(directory=tmp_path)
    test_app = FastAPI()
    test_app.mount("/media", safe_app, name="media")
    return TestClient(test_app)


def test_safe_static_files_blocks_database_and_sensitive_files(client, tmp_path):
    """Direct or indirect requests for database files, environment secrets, and scripts return 404."""
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()

    # Create dummy database and secrets files
    (tmp_path / "birdwatcher.db").write_bytes(b"SQLite format 3\x00")
    (tmp_path / "birdwatcher.db-wal").write_bytes(b"wal content")
    (tmp_path / ".env").write_text("SECRET_KEY=supersecret")
    (tmp_path / "settings.json").write_text('{"secret": "val"}')
    (crops_dir / "exploit.py").write_text("import os; os.system('id')")

    # Database blocked
    assert client.get("/media/birdwatcher.db").status_code == 404
    assert client.get("/media/birdwatcher.db-wal").status_code == 404
    assert client.get("/media/birdwatcher.sqlite").status_code == 404

    # Secrets and source code blocked
    assert client.get("/media/.env").status_code == 404
    assert client.get("/media/settings.json").status_code == 404
    assert client.get("/media/crops/exploit.py").status_code == 404

    # Non-allowed root file
    assert client.get("/media/random.txt").status_code == 404


def test_safe_static_files_blocks_unauthorized_directories(client, tmp_path):
    """Only approved directories (crops, clips, heatmaps, calibration, frames) are accessible."""
    forbidden_dir = tmp_path / "private_folder"
    forbidden_dir.mkdir()
    (forbidden_dir / "photo.jpg").write_bytes(b"\xff\xd8\xff dummy jpeg")

    assert client.get("/media/private_folder/photo.jpg").status_code == 404


def test_safe_static_files_allows_valid_media(client, tmp_path):
    """Allowed directories with safe extensions (e.g. .jpg, .mp4) serve normally."""
    crops_dir = tmp_path / "crops"
    clips_dir = tmp_path / "clips"
    crops_dir.mkdir()
    clips_dir.mkdir()

    valid_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    valid_mp4 = b"\x00\x00\x00\x18ftypmp42"

    (crops_dir / "v00000001_t0001.jpg").write_bytes(valid_jpg)
    (clips_dir / "Birdfeeder_test.mp4").write_bytes(valid_mp4)

    res_jpg = client.get("/media/crops/v00000001_t0001.jpg")
    assert res_jpg.status_code == 200
    assert res_jpg.content == valid_jpg

    res_mp4 = client.get("/media/clips/Birdfeeder_test.mp4")
    assert res_mp4.status_code == 200
    assert res_mp4.content == valid_mp4
