"""Unit tests for src/poi.py — the 27 candidate POIs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import poi  # noqa: E402


def test_count_is_27():
    assert len(poi.CANDIDATE_POIS) == 27


def test_fields_valid():
    for en, zh, lat, lon, kind in poi.CANDIDATE_POIS:
        assert en and zh                       # both names present
        assert 47.34 < lat < 47.40             # inside greater Zurich
        assert 8.50 < lon < 8.58
        assert kind in poi.KIND_ICON           # every kind has an icon


def test_names_unique():
    names = [p[0] for p in poi.CANDIDATE_POIS]
    assert len(names) == len(set(names))
