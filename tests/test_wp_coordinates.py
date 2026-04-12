"""
Tests for create_wordpress_sizes_with_pixelation coordinate mapping (bug 3).

Strategy: create a wide RED image (400x200) that when resized non-crop to 200x200
will be letterboxed with a 50px top/bottom white border. The canvas will be:
  rows 0-49:    white padding
  rows 50-149:  red content
  rows 150-199: white padding

A detection at [100, 50, 200, 100] in original coords is scaled with 0.5 and
should land at canvas coords (50, 75, 100, 50) after adding the paste offset of 50.

With the BUG (offset missing): pixelation region spans canvas y=23..77,
which mixes white padding rows with red content rows during the pixelation
downscale. The block covering canvas rows ~41-58 is averaged into a pinkish
colour (~127 on the blue-green channels).

With the FIX (offset applied): pixelation region is y=73..127, entirely
inside the red content zone. The white padding rows 0-49 are untouched and
stay pure white (255, 255, 255).

The test therefore asserts that canvas row 45 (inside the padding zone, which
the buggy code wrongly includes in the pixelation region) is PURE WHITE after
the function runs.
"""

import os
import numpy as np
import cv2
import pytest


def _make_red_image(width, height, tmp_path, name="input.png"):
    """Solid red PNG image (lossless, so pixelation boundaries stay sharp)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # BGR: red channel
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    return path


def test_noncrop_paste_offset_applied(tmp_path, monkeypatch):
    """
    Non-crop resize of a 400x200 RED image to 200x200:
      scale_factor = min(200/400, 200/200) = 0.5
      new_w=200, new_h=100
      paste_y=50  <-- this offset must be added to scaled detection y coords

    Detection box [100, 50, 200, 100] in original coords:
      scaled (buggy):  y=25  -> pixelation at canvas y ≈ 23..77
      scaled (correct): y=75 -> pixelation at canvas y ≈ 73..127

    With the BUG: the pixelation region spans white-padding rows + red-content
    rows, so the downscale blends them.  Canvas row 45 ends up pinkish
    (~[127, 127, 255] BGR) rather than pure white.

    With the FIX: rows 0-49 are never touched; canvas row 45 stays (255,255,255).
    """
    import main as m

    # Patch WORDPRESS_SIZES to only create one simple 200x200 non-crop size
    monkeypatch.setattr(m, "WORDPRESS_SIZES", {"test-nc": (200, 200, False)})

    input_path = _make_red_image(400, 200, tmp_path)
    detection = {
        "box": [100, 50, 200, 100],
        "score": 0.9,
        "class": "FEMALE_BREAST_EXPOSED",
    }

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)   # wp-content/uploads will be created relative to here
    try:
        m.create_wordpress_sizes_with_pixelation(
            input_path, [detection], "output", image_type=None, pixel_size=15
        )
    finally:
        os.chdir(orig_cwd)

    result_path = str(tmp_path / "wp-content" / "uploads" / "output-200x200.png")
    assert os.path.exists(result_path), f"Output file not created: {result_path}"

    result_img = cv2.imread(result_path)

    # Canvas row 45 is inside the top white-padding zone (rows 0-49).
    # The paste offset must NOT cause the pixelation to bleed into this zone.
    # With the bug the downscale of a mixed white+red region produces a pinkish
    # value (~127 on the blue-green channels) here instead of pure white.
    padding_row_pixel = result_img[45, 100]   # BGR
    assert all(v > 245 for v in padding_row_pixel), (
        f"Canvas row 45 is {padding_row_pixel} — expected pure white (255,255,255). "
        "The paste offset was not applied; the pixelation region wrongly overlaps the "
        "white padding, blending white and red content into a pinkish colour."
    )
