"""Unit tests for src/geo_check.py — pure scan-row -> (gps, conf)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import geo_check as g  # noqa: E402


def test_from_scan_with_guess_match():
    pois_map = {"Bahnhofstrasse": {"name": "Bahnhofstrasse",
                                   "lat": 47.37, "lon": 8.54}}
    row = {"guess": "Bahnhofstrasse", "confidence": "High",
           "reasoning": "luxury watch shops",
           "matched": [{"source": "guess", "osm_name": "Bahnhofstrasse"},
                       {"source": "visible", "osm_name": "Cartier"}]}
    r = g.geo_check_from_scan(row, pois_map)
    assert r["gps"] == (47.37, 8.54)
    assert r["confidence"] == "high"             # lower-cased
    assert r["place_name"] == "Bahnhofstrasse"
    assert r["reasoning"] == "luxury watch shops"


def test_from_scan_no_guess_means_no_gps():
    # only a 'visible' match -> we still have no inferred LOCATION
    pois_map = {"Cartier": {"name": "Cartier", "lat": 47.37, "lon": 8.54}}
    row = {"guess": "", "confidence": "low", "reasoning": "",
           "matched": [{"source": "visible", "osm_name": "Cartier"}]}
    r = g.geo_check_from_scan(row, pois_map)
    assert r["gps"] is None
    assert r["place_name"] == ""


def test_from_scan_guess_not_in_pois_map():
    row = {"guess": "Imaginary Plaza", "confidence": "medium",
           "reasoning": "",
           "matched": [{"source": "guess", "osm_name": "Imaginary Plaza"}]}
    r = g.geo_check_from_scan(row, {})           # POI table empty
    assert r["gps"] is None
    assert r["confidence"] == ""                 # drops when unresolved


def test_from_scan_empty_row():
    r = g.geo_check_from_scan({}, {})
    assert r == {"gps": None, "confidence": "",
                 "place_name": "", "reasoning": ""}
