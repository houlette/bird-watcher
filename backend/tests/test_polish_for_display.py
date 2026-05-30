"""Tests for the user-feed image polish (CLAHE + unsharp) in pipeline.process.

The polish only affects the JPEG written to disk for the feed; the classifier
input goes through a separate path in pipeline.classify. These tests verify
the function preserves shape/type, lifts contrast on flat images (CLAHE),
and increases high-frequency content vs the unprocessed input (sharpening).
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


def test_polish_increases_high_frequency_content():
    """Unsharp mask's job is to raise the magnitude of the Laplacian (edge
    response). Use mid-tone stripes (not 0/255 saturation extremes — those
    are clamped by the kernel before sharpening can act on them)."""
    img = np.full((200, 200, 3), 100, dtype=np.uint8)
    img[::4, :] = 160         # mid-tone horizontal stripes
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    polished_gray = cv2.cvtColor(_polish_for_display(img), cv2.COLOR_BGR2GRAY)
    in_lap = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    out_lap = float(cv2.Laplacian(polished_gray, cv2.CV_64F).var())
    assert out_lap > in_lap


def test_polish_widens_dynamic_range_on_dim_input():
    """CLAHE component should spread the L-channel histogram on a flat,
    narrow-range crop."""
    img = np.full((150, 150, 3), 80, dtype=np.uint8)
    img[40:110, 40:110] = 90    # barely-visible patch
    in_l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    out_l = cv2.cvtColor(_polish_for_display(img), cv2.COLOR_BGR2LAB)[:, :, 0]
    assert out_l.std() > in_l.std()


def test_polish_does_not_introduce_color_cast_on_neutral_gray():
    """LAB-channel CLAHE leaves a, b alone; sharpening on a uniform image is
    a no-op. A fully neutral gray crop should stay neutral."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    out_lab = cv2.cvtColor(_polish_for_display(img), cv2.COLOR_BGR2LAB)
    # Tolerance of 2 because the unsharp-mask's gaussian/addWeighted can
    # nudge by sub-integer rounding even with no edge content.
    assert abs(int(out_lab[:, :, 1].mean()) - 128) <= 2
    assert abs(int(out_lab[:, :, 2].mean()) - 128) <= 2
