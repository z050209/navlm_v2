"""Unit tests for src/extract_frames.py — the quality-filter functions.

Uses synthetic images, so it runs without any real video or ffmpeg.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                       # noqa: E402
from src import extract_frames as ef  # noqa: E402


def _checkerboard(n=256, cell=16):
    """A high-contrast checkerboard — many edges, high Laplacian variance."""
    a = np.zeros((n, n), np.uint8)
    for i in range(0, n, cell):
        for j in range(0, n, cell):
            if (i // cell + j // cell) % 2 == 0:
                a[i:i + cell, j:j + cell] = 255
    return np.stack([a] * 3, axis=-1)


def test_quality_metrics_sharp_vs_flat(tmp_path):
    sharp = tmp_path / "sharp.jpg"
    flat = tmp_path / "flat.jpg"
    Image.fromarray(_checkerboard()).save(sharp)
    Image.fromarray(np.full((256, 256, 3), 128, np.uint8)).save(flat)

    blur_sharp, _ = ef.quality_metrics(sharp)
    blur_flat, _ = ef.quality_metrics(flat)

    assert blur_sharp > blur_flat
    assert blur_sharp >= config.BLUR_MIN_VAR      # checkerboard passes
    assert blur_flat < config.BLUR_MIN_VAR        # flat image fails blur gate


def test_quality_metrics_luma(tmp_path):
    dark = tmp_path / "dark.jpg"
    bright = tmp_path / "bright.jpg"
    Image.fromarray(np.full((64, 64, 3), 5, np.uint8)).save(dark)
    Image.fromarray(np.full((64, 64, 3), 250, np.uint8)).save(bright)

    assert ef.quality_metrics(dark)[1] < config.EXPOSURE_DARK
    assert ef.quality_metrics(bright)[1] > config.EXPOSURE_BRIGHT


def test_quality_metrics_missing_file():
    assert ef.quality_metrics(Path("does_not_exist.jpg")) == (0.0, 0.0)


def test_passes_quality():
    assert ef.passes_quality(500.0, 128) == (True, "")
    assert ef.passes_quality(10.0, 128) == (False, "blur")     # below BLUR_MIN_VAR
    assert ef.passes_quality(500.0, 5) == (False, "exposure")  # too dark
    assert ef.passes_quality(500.0, 250) == (False, "exposure")  # blown out


def test_discover_videos_returns_name_path_pairs():
    found = ef.discover_videos()
    assert isinstance(found, list)
    for entry in found:
        name, path = entry
        assert isinstance(name, str) and name
        assert str(path).lower().endswith(".mp4")
