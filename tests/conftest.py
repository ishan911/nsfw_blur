"""Shared pytest fixtures for blurapi test suite."""

import os
import tempfile
import numpy as np
import cv2
import pytest


@pytest.fixture
def solid_image_path(tmp_path):
    """Write a 200x200 solid-colour JPEG to a temp directory, return its path."""
    img = np.full((200, 200, 3), (120, 80, 60), dtype=np.uint8)
    path = str(tmp_path / "test_input.jpg")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def output_path(tmp_path):
    """Return a path for output images inside the temp directory."""
    return str(tmp_path / "test_output.jpg")
