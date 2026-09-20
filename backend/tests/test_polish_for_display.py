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


def test_chroma_guided_filter_edge_cases():
    from pipeline.process import _chroma_guided_filter

    # Empty
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert _chroma_guided_filter(empty).size == 0
    # None
    assert _chroma_guided_filter(None) is None
    # Tiny (smaller than filter window)
    tiny = np.ones((3, 3, 3), dtype=np.uint8) * 100
    assert _chroma_guided_filter(tiny).shape == (3, 3, 3)
    # Uniform solid color is preserved within color space roundtrip precision (<= 1 LSB)
    solid = (np.ones((40, 40, 3), dtype=np.uint8) * [40, 80, 200]).astype(np.uint8)
    out_solid = _chroma_guided_filter(solid)
    np.testing.assert_allclose(solid, out_solid, atol=1)


def test_chroma_guided_filter_denoises_color_channels():
    """Chroma noise on a flat region should be significantly smoothed."""
    from pipeline.process import _chroma_guided_filter

    img = np.full((80, 80, 3), 128, dtype=np.uint8)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    np.random.seed(123)
    ycrcb[:, :, 1] += np.random.normal(0, 15, (80, 80))  # Cr noise
    ycrcb[:, :, 2] += np.random.normal(0, 15, (80, 80))  # Cb noise
    noisy_bgr = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    noisy_cr_std = cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 1].std()
    filtered_bgr = _chroma_guided_filter(noisy_bgr)
    filtered_cr_std = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 1].std()

    # Noise std should be reduced by more than 50%
    assert filtered_cr_std < noisy_cr_std * 0.5


def test_chroma_filter_toggled_by_settings(monkeypatch):
    """When chroma_filter_enabled is False, _polish_for_display skips chroma filtering."""
    from settings import settings

    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    img[25, 25] = [200, 50, 50]

    monkeypatch.setattr(settings, "chroma_filter_enabled", False)
    out_disabled = _polish_for_display(img)

    monkeypatch.setattr(settings, "chroma_filter_enabled", True)
    out_enabled = _polish_for_display(img)

    # Output with filter enabled should differ from disabled due to smoothing the isolated chroma spike
    assert not np.array_equal(out_disabled, out_enabled)


def test_render_crop_variant_combinations():
    """Verify render_crop_variant correctly handles raw passthrough and flags."""
    from pipeline.process import render_crop_variant, _polish_for_display

    img = np.random.randint(40, 200, (60, 60, 3), dtype=np.uint8)

    # All false: returns unchanged raw pixels
    raw = render_crop_variant(img, chroma=False, clahe=False, sharpen=False)
    np.testing.assert_array_equal(img, raw)

    # All true matches _polish_for_display
    full = render_crop_variant(img, chroma=True, clahe=True, mertens=True, sharpen=True)
    polish = _polish_for_display(img)
    np.testing.assert_array_equal(full, polish)

    # Individual flags produce differing images
    chroma_only = render_crop_variant(img, chroma=True, clahe=False, mertens=False, sharpen=False)
    clahe_only = render_crop_variant(img, chroma=False, clahe=True, mertens=False, sharpen=False)
    mertens_only = render_crop_variant(img, chroma=False, clahe=False, mertens=True, sharpen=False)
    sharpen_only = render_crop_variant(img, chroma=False, clahe=False, mertens=False, sharpen=True)

    assert not np.array_equal(chroma_only, img)
    assert not np.array_equal(clahe_only, img)
    assert not np.array_equal(mertens_only, img)
    assert not np.array_equal(chroma_only, clahe_only)


def test_mertens_exposure_fusion_edge_cases():
    """Verify Mertens fusion handles empty, None, and small inputs safely."""
    from pipeline.process import _mertens_exposure_fusion

    assert _mertens_exposure_fusion(None) is None
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert _mertens_exposure_fusion(empty).size == 0

    tiny = np.full((2, 2, 3), 128, dtype=np.uint8)
    out_tiny = _mertens_exposure_fusion(tiny)
    assert out_tiny.shape == (2, 2, 3)

    solid = np.full((40, 40, 3), 128, dtype=np.uint8)
    out_solid = _mertens_exposure_fusion(solid)
    assert out_solid.shape == (40, 40, 3)
    # Uniform gray should be preserved near midtone
    assert abs(int(out_solid[0, 0, 0]) - 128) < 10


def test_mertens_exposure_fusion_recovers_contrast():
    """Mertens should lift dark shadows while preserving highlights without clipping."""
    from pipeline.process import _mertens_exposure_fusion

    # High-contrast synthetic image: deep shadow on left, bright highlight on right
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[:, :30] = [20, 25, 20]     # deep plumage shadow
    img[:, 30:] = [230, 225, 220]  # bright sunlit feather highlight

    fused = _mertens_exposure_fusion(img)

    # Shadow should be lifted above raw
    assert fused[:, :30].mean() > img[:, :30].mean()
    # Highlight should remain unclipped (< 255)
    assert fused[:, 30:].max() <= 250


