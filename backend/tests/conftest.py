"""Shared pytest fixtures.

Two cross-cutting concerns:

  1. Tests must not pick up the host machine's real `yard_priors.json`. We
     point the calibration path at a non-existent file by default so each
     test starts from a clean uncalibrated state. Tests that want a
     calibration in scope can override via their own monkeypatch.

  2. In-memory SQLite engines used as test fixtures need
     `check_same_thread=False` so FastAPI's TestClient (which runs the
     handler in a worker thread) doesn't crash on connection cleanup.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import calibration


@pytest.fixture(autouse=True)
def _no_calibration_by_default(monkeypatch, tmp_path) -> None:
    """Point the calibration loader at a non-existent path for every test.

    Individual tests can re-monkeypatch CALIBRATION_PATH to a real file
    inside their own tmp_path to exercise the calibrated code path.
    """
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", tmp_path / "no_calibration.json")
    calibration.reset_cache_for_tests()
    yield
    calibration.reset_cache_for_tests()
