"""Unit tests for src/poi_scan.py — OSM-tag tiering, parsing, matching."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import poi_scan as ps  # noqa: E402


def test_poi_tier_by_osm_tag():
    # L1 — landmark categories
    assert ps.poi_tier("tourism=attraction") == 1
    assert ps.poi_tier("amenity=place_of_worship") == 1
    assert ps.poi_tier("railway=station") == 1
    # L2 — supporting POIs
    assert ps.poi_tier("tourism=museum") == 2
    assert ps.poi_tier("man_made=bridge") == 2
    assert ps.poi_tier("highway=primary") == 2          # named street
    # L3 — key-only fallback / unknown
    assert ps.poi_tier("amenity=cafe") == 3             # amenity fallback -> 3
    assert ps.poi_tier("") == 3


def test_parse_names_variants():
    out = ps.parse_names("Hauptbahnhof | Zurich Main Station\nLimmat\nnone")
    assert out == [["Hauptbahnhof", "Zurich Main Station"], ["Limmat"]]
    assert ps.parse_names("- Grossmünster") == [["Grossmünster"]]
    assert ps.parse_names("") == []


def test_match_names_variant_fallback():
    osm = [
        {"name": "Hauptbahnhof", "aliases": [], "osm_kind": "railway=station"},
        {"name": "Grossmünster", "aliases": [],
         "osm_kind": "amenity=place_of_worship"},
    ]
    places = [["Zurich Main Station", "Hauptbahnhof"], ["Nowhere"]]
    matched, unmatched = ps.match_names(places, osm)
    assert len(matched) == 1
    assert matched[0]["osm_name"] == "Hauptbahnhof"
    assert matched[0]["matched_name"] == "Hauptbahnhof"
    assert matched[0]["tier"] == 1                      # railway=station -> L1
    assert unmatched == [["Nowhere"]]


def test_match_names_diacritic():
    osm = [{"name": "Grossmünster", "aliases": [],
            "osm_kind": "amenity=place_of_worship"}]
    matched, _ = ps.match_names([["Grossmunster"]], osm)   # no umlaut
    assert matched and matched[0]["osm_name"] == "Grossmünster"
    assert matched[0]["tier"] == 1
