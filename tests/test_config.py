"""Unit tests for config.py — paths, bbox, video-name mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def test_dataset_name_known_titles():
    assert config.dataset_name(
        "Zurich in Summer Hidden Streets [4K HDR 60FPS].mp4") == "hidden_streets"
    assert config.dataset_name(
        "ZURICH, Switzerland - 4K 60fps.mp4") == "zurich_main"
    assert config.dataset_name(
        "Switzerland Zurich Bahnhofstrasse Walking tour 4K 60fps HDR.mp4"
    ) == "bahnhofstrasse"
    assert config.dataset_name(
        "ZURICH The Most Elegant City in Europe 4K.mp4") == "most_elegant"


def test_dataset_name_fallback_sanitizes():
    name = config.dataset_name("Totally Unknown Clip!!.mp4")
    assert name and " " not in name and name == name.lower()


def test_sv_bbox_has_margin_around_poi_bbox():
    w, s, e, n = config.POI_BBOX
    W, S, E, N = config.SV_BBOX
    assert w < e and s < n               # POI bbox well-formed
    assert W < w and S < s and E > e and N > n   # SV bbox strictly larger


def test_paths_derive_from_data_root():
    assert config.VIDEOS_DIR.name == "videos"
    assert str(config.FRAMES_DIR).startswith(str(config.DATA_ROOT))
    assert config.HOLDOUT_VIDEO in config.VIDEOS.values()


def test_ffmpeg_resolved():
    assert config.FFMPEG                 # non-empty path or "ffmpeg"
    assert config.FFMPEG_DIR
