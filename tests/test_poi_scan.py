"""Unit tests for src/poi_scan.py — OSM-tag tiering, JSON parsing,
matching of the inference-based scan."""

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


def test_parse_scan_json():
    txt = ('{"visible": ["Cartier"], '
           '"guess": "Bahnhofstrasse | Bahnhof Street", '
           '"confidence": "High", "reasoning": "luxury watch shops"}')
    d = ps.parse_scan(txt)
    assert d["visible"] == [["Cartier"]]
    assert d["guess"] == ["Bahnhofstrasse", "Bahnhof Street"]
    assert d["confidence"] == "high"                    # lower-cased
    assert d["reasoning"] == "luxury watch shops"


def test_parse_scan_fenced_and_prose():
    txt = ('Sure, here is the answer:\n```json\n'
           '{"visible": [], "guess": "Limmat", '
           '"confidence": "low", "reasoning": "river"}\n```')
    d = ps.parse_scan(txt)
    assert d["guess"] == ["Limmat"]
    assert d["visible"] == []
    assert d["confidence"] == "low"


def test_parse_scan_garbage_is_empty():
    assert ps.parse_scan("sorry, I cannot tell") == ps.EMPTY_SCAN
    assert ps.parse_scan("") == ps.EMPTY_SCAN
    assert ps.parse_scan("{not valid json}") == ps.EMPTY_SCAN


def test_match_scan_visible_and_guess():
    osm = [
        {"name": "Hauptbahnhof", "aliases": [], "osm_kind": "railway=station",
         "kind_label": "a railway station"},
        {"name": "Bahnhofstrasse", "aliases": [], "osm_kind": "highway=primary",
         "kind_label": "a street"},
    ]
    parsed = {"visible": [["Zurich Main Station", "Hauptbahnhof"]],
              "guess": ["Bahnhofstrasse"], "confidence": "high",
              "reasoning": ""}
    matched, unmatched = ps.match_scan(parsed, osm)
    assert len(matched) == 2
    by = {m["osm_name"]: m for m in matched}
    assert by["Hauptbahnhof"]["source"] == "visible"
    assert by["Hauptbahnhof"]["tier"] == 1              # railway=station
    assert by["Hauptbahnhof"]["matched_name"] == "Hauptbahnhof"  # variant 2
    assert by["Bahnhofstrasse"]["source"] == "guess"
    assert by["Bahnhofstrasse"]["tier"] == 2            # highway -> L2
    assert unmatched == []


def test_match_scan_unmatched_and_diacritic():
    osm = [{"name": "Grossmünster", "aliases": [],
            "osm_kind": "amenity=place_of_worship", "kind_label": "a church"}]
    parsed = {"visible": [["Nowhere Plaza"]], "guess": ["Grossmunster"],
              "confidence": "medium", "reasoning": ""}    # guess: no umlaut
    matched, unmatched = ps.match_scan(parsed, osm)
    assert len(matched) == 1
    assert matched[0]["osm_name"] == "Grossmünster"      # diacritic-folded
    assert matched[0]["source"] == "guess"
    assert matched[0]["tier"] == 1
    assert unmatched == [["Nowhere Plaza"]]


def test_match_scan_empty():
    matched, unmatched = ps.match_scan(ps.EMPTY_SCAN, [])
    assert matched == [] and unmatched == []


def test_downscaled_resizes(tmp_path):
    import os
    from PIL import Image
    big = tmp_path / "big.jpg"
    Image.new("RGB", (2400, 1600), "white").save(big)
    small = ps._downscaled(big, max_px=1024)
    try:
        w, h = Image.open(small).size
        assert max(w, h) <= 1024
    finally:
        os.unlink(small)
