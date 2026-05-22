"""Unit tests for src/poi_scan.py — tiering, variant parsing, matching."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import poi_scan as ps  # noqa: E402


def test_poi_tier():
    assert ps.poi_tier("Grossmünster") == 1                  # iconic
    assert ps.poi_tier("Helmhaus Museum", "a museum") == 2   # mid
    assert ps.poi_tier("Random Cafe", "a cafe") == 3         # other


def test_parse_names_variants():
    out = ps.parse_names("Hauptbahnhof | Zurich Main Station\nLimmat\nnone")
    assert out == [["Hauptbahnhof", "Zurich Main Station"], ["Limmat"]]
    assert ps.parse_names("- Grossmünster") == [["Grossmünster"]]
    assert ps.parse_names("") == []
    assert ps.parse_names("none") == []


def test_match_names_variant_fallback():
    osm = [
        {"name": "Hauptbahnhof", "aliases": [], "kind_label": "a station"},
        {"name": "Grossmünster", "aliases": [], "kind_label": "a church"},
    ]
    # Gemini's first variant misses; the second (the OSM name) hits
    places = [["Zurich Main Station", "Hauptbahnhof"], ["Nowhere"]]
    matched, unmatched = ps.match_names(places, osm)
    assert len(matched) == 1
    assert matched[0]["osm_name"] == "Hauptbahnhof"
    assert matched[0]["matched_name"] == "Hauptbahnhof"      # variant that hit
    assert unmatched == [["Nowhere"]]


def test_match_names_diacritic():
    osm = [{"name": "Grossmünster", "aliases": [], "kind_label": "a church"}]
    matched, _ = ps.match_names([["Grossmunster"]], osm)     # no umlaut
    assert matched and matched[0]["osm_name"] == "Grossmünster"
