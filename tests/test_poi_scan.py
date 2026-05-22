"""Unit tests for src/poi_scan.py — tiering, parsing, OSM matching."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import poi_scan as ps  # noqa: E402


def test_poi_tier():
    assert ps.poi_tier("Grossmünster") == 1                  # iconic
    assert ps.poi_tier("Helmhaus Museum", "a museum") == 2   # mid
    assert ps.poi_tier("Random Cafe", "a cafe") == 3         # other


def test_parse_names():
    assert ps.parse_names("Grossmünster\nBahnhofstrasse") == [
        "Grossmünster", "Bahnhofstrasse"]
    assert ps.parse_names("- Limmat\n* Lake Zurich") == [
        "Limmat", "Lake Zurich"]
    assert ps.parse_names("none") == []
    assert ps.parse_names("") == []


def test_match_names():
    osm = [
        {"name": "Grossmünster", "aliases": [], "kind_label": "a church"},
        {"name": "Polyterrasse", "aliases": ["ETH"], "kind_label": "a viewpoint"},
    ]
    matched, unmatched = ps.match_names(["Grossmünster", "ETH", "Nowhere"], osm)
    by = {m["osm_name"]: m for m in matched}
    assert "Grossmünster" in by and "Polyterrasse" in by   # ETH -> Polyterrasse
    assert "Nowhere" in unmatched
    assert by["Grossmünster"]["tier"] == 1
    assert by["Polyterrasse"]["raw"] == "ETH"              # raw name kept
