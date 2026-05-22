"""Unit tests for src/pois.py — OSM name cleaning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import pois  # noqa: E402


def test_clean_name_keeps_valid():
    assert pois.clean_name("Grossmünster") == "Grossmünster"
    assert pois.clean_name("  Bahnhofstrasse  ") == "Bahnhofstrasse"


def test_clean_name_drops_invalid():
    assert pois.clean_name("") is None
    assert pois.clean_name(None) is None
    assert pois.clean_name("unknown") is None          # blocklisted
    assert pois.clean_name("ab") is None               # too short
    assert pois.clean_name("x" * 50) is None           # too long
    assert pois.clean_name("lowercase start") is None  # not capitalised
