"""Unit tests for yard calibration loading + fallback behavior.

Cover three paths:
  - No calibration file present → both functions return None / sentinel so
    callers know to use their hard-coded fallbacks.
  - Well-formed calibration file → values flow through.
  - Malformed JSON → graceful no-op (logged warning, callers fall back).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from pipeline import calibration


@pytest.fixture(autouse=True)
def reset_calibration_cache():
    calibration.reset_cache_for_tests()
    yield
    calibration.reset_cache_for_tests()


def _write_calibration(path, species: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": {"yearly_years": [2025], "daily_days": 365},
        "species": species,
    }))


def test_no_file_means_no_allowlist(monkeypatch, tmp_path):
    """When the calibration file doesn't exist, callers get None to fall back."""
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", tmp_path / "missing.json")
    assert calibration.is_calibrated() is False
    assert calibration.get_allowlist() is None


def test_no_file_means_neutral_seasonal_multiplier(monkeypatch, tmp_path):
    """When the calibration file doesn't exist, monthly multiplier returns None
    so fuse.py knows to use its hand-coded table."""
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", tmp_path / "missing.json")
    assert calibration.get_monthly_multiplier("Northern Cardinal", 5) is None


def test_loaded_calibration_returns_allowlist(monkeypatch, tmp_path):
    cal_path = tmp_path / "yard_priors.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", cal_path)
    _write_calibration(cal_path, {
        "Northern Cardinal": {"total": 800, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
        "Carolina Wren": {"total": 250, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
        # Below threshold — excluded.
        "Some Vagrant": {"total": 1, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
    })
    allowlist = calibration.get_allowlist()
    assert allowlist == {"Northern Cardinal", "Carolina Wren"}


def test_seasonal_multiplier_boosts_concentrated_month(monkeypatch, tmp_path):
    """A species detected 60% in February gets a strong February boost."""
    cal_path = tmp_path / "yard_priors.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", cal_path)
    pct = {str(m): 0.0 for m in range(1, 13)}
    pct["2"] = 0.6
    pct["3"] = 0.4
    _write_calibration(cal_path, {
        "Snow Bunting": {"total": 50, "monthly_pct": pct},
    })
    # 0.6 × 12 = 7.2, clamped to MAX_SEASONAL_MULTIPLIER=4.0
    assert calibration.get_monthly_multiplier("Snow Bunting", 2) == calibration.MAX_SEASONAL_MULTIPLIER
    # 0.4 × 12 = 4.8 also clamped
    assert calibration.get_monthly_multiplier("Snow Bunting", 3) == calibration.MAX_SEASONAL_MULTIPLIER
    # In an off-month, 0.0 × 12 = 0.0, clamped up to MIN_SEASONAL_MULTIPLIER
    assert calibration.get_monthly_multiplier("Snow Bunting", 7) == calibration.MIN_SEASONAL_MULTIPLIER


def test_seasonal_multiplier_neutral_for_unknown_species(monkeypatch, tmp_path):
    """A species not in the calibration file still returns 1.0 (neutral)."""
    cal_path = tmp_path / "yard_priors.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", cal_path)
    _write_calibration(cal_path, {
        "Northern Cardinal": {"total": 800, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
    })
    assert calibration.get_monthly_multiplier("Some Other Bird", 5) == 1.0


def test_malformed_json_falls_back(monkeypatch, tmp_path):
    cal_path = tmp_path / "yard_priors.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", cal_path)
    cal_path.write_text("{ not valid json")
    assert calibration.is_calibrated() is False
    assert calibration.get_allowlist() is None


def test_wrong_shape_falls_back(monkeypatch, tmp_path):
    cal_path = tmp_path / "yard_priors.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", cal_path)
    cal_path.write_text(json.dumps({"unrelated": "data"}))
    assert calibration.is_calibrated() is False
    assert calibration.get_allowlist() is None


def test_fuse_uses_calibration_when_available(monkeypatch, tmp_path):
    """Smoke test: fuse._seasonal_multiplier picks the calibrated value, not the hand-coded one."""
    from pipeline import fuse

    cal_path = tmp_path / "yard_priors.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", cal_path)
    # Set a heavily-skewed July pattern for a species also in the hand-coded table.
    pct = {str(m): 0.0 for m in range(1, 13)}
    pct["7"] = 1.0
    _write_calibration(cal_path, {
        "Dark-eyed Junco": {"total": 100, "monthly_pct": pct},
    })
    # Hand-coded table has Dark-eyed Junco at 0.1 in July (summer-rare);
    # calibration would yield 1.0 × 12 = 12.0, clamped to 4.0.
    assert fuse._seasonal_multiplier("Dark-eyed Junco", 7) == calibration.MAX_SEASONAL_MULTIPLIER


def test_fuse_falls_back_when_uncalibrated(monkeypatch, tmp_path):
    """When no calibration file exists, fuse falls back to the hand-coded table."""
    from pipeline import fuse

    monkeypatch.setattr(calibration, "CALIBRATION_PATH", tmp_path / "missing.json")
    # The hand-coded summer Dark-eyed Junco multiplier is 0.1.
    assert fuse._seasonal_multiplier("Dark-eyed Junco", 7) == 0.1
