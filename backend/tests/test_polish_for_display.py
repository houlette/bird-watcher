"""Tests for the user-feed image polish (CLAHE) in pipeline.process.

The polish only affects the JPEG written to disk for the feed; the classifier
input goes through a separate path in pipeline.classify. Sharpening was
removed after several rounds of user feedback that it read as crunchy on
feather edges — these tests now verify only CLAHE behavior (shape/dtype
preservation, dynamic-range widening on dim input, no color cast).
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
