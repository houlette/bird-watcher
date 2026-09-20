"""Tests for the user-feed image polish (CLAHE + edge-aware sharpener) in pipeline.process.

The polish affects the JPEG written to disk for the feed; the classifier
input goes through its own preprocessing in pipeline.classify. We test:
  - shape and dtype preservation
  - dynamic-range widening via CLAHE on dim input
  - color neutrality on neutral gray
  - edge-aware sharpening on blurry input (lifting high-frequency variance)
  - gating on already-crisp input (preventing oversharpening)
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline.process import _polish_for_display


def test_polish_preserves_shape_and_dtype():
    img = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    out = _polish_for_display(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_polish_widens_dynamic_range_on_dim_input():
    """CLAHE component should spread the L-channel histogram on a flat,
    narrow-range crop."""
    img = np.full((150, 150, 3), 80, dtype=np.uint8)
    img[40:110, 40:110] = 90    # barely-visible patch
    in_l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    out_l = cv2.cvtColor(_polish_for_display(img), cv2.COLOR_BGR2LAB)[:, :, 0]
    assert out_l.std() > in_l.std()


def test_polish_does_not_introduce_color_cast_on_neutral_gray():
    """LAB-channel CLAHE leaves a, b alone. A fully neutral gray crop
    stays neutral."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    out_lab = cv2.cvtColor(_polish_for_display(img), cv2.COLOR_BGR2LAB)
    assert abs(int(out_lab[:, :, 1].mean()) - 128) <= 1
    assert abs(int(out_lab[:, :, 2].mean()) - 128) <= 1


def test_polish_sharpens_blurry_input():
    """Soft/blurry images should gain high-frequency detail after polish."""
    # Create an image with soft edges
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(base, (50, 50), 30, (200, 200, 200), -1)
    blurry = cv2.GaussianBlur(base, (9, 9), 2.5)

    var_before = cv2.Laplacian(cv2.cvtColor(blurry, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    polished = _polish_for_display(blurry)
    var_after = cv2.Laplacian(cv2.cvtColor(polished, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    assert var_after > var_before


def test_polish_skips_sharpening_on_already_crisp_input(monkeypatch):
    """High-contrast/crisp crops above the cutoff threshold should skip sharpening."""
    from pipeline import process

    # Force cutoff threshold low to verify gating
    monkeypatch.setattr(process, "_SHARPEN_CUTOFF_VAR", 50.0)

    # Sharp checkered pattern with high variance
    sharp = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp[::2, ::2] = 255
    sharp[1::2, 1::2] = 255

    # With max_amount = 0 (CLAHE only), output should equal output when gated
    monkeypatch.setattr(process, "_SHARPEN_MAX_AMOUNT", 0.0)
    clahe_only = _polish_for_display(sharp)

    # Restore max_amount; because variance >> cutoff, it should still equal clahe_only
    monkeypatch.setattr(process, "_SHARPEN_MAX_AMOUNT", 0.4)
    gated = _polish_for_display(sharp)

    np.testing.assert_array_equal(clahe_only, gated)
