"""Unit tests for src/spatial.py — POI index + nearest + name match."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import spatial as sp  # noqa: E402


def test_nearest_returns_self_for_zero_distance():
    pois = [{"name": "A", "aliases": [],
             "geometry": "POINT (8.54 47.37)",
             "lat": 47.37, "lon": 8.54}]
    idx = sp.build_poi_index(pois)
    p, d = sp.nearest_poi_m(47.37, 8.54, idx)
    assert p["name"] == "A"
    assert d < 1.0


def test_nearest_picks_closer_point_poi():
    pois = [
        {"name": "Far", "aliases": [],
         "geometry": "POINT (8.60 47.40)", "lat": 47.40, "lon": 8.60},
        {"name": "Close", "aliases": [],
         "geometry": "POINT (8.541 47.371)",
         "lat": 47.371, "lon": 8.541},
    ]
    idx = sp.build_poi_index(pois)
    p, _ = sp.nearest_poi_m(47.37, 8.54, idx)
    assert p["name"] == "Close"


def test_long_street_matched_by_point_to_line_not_centroid():
    """A street running W-E across central Zurich. A point ON the
    street far from its centroid must still resolve to it."""
    pois = [
        # ~1.3 km horizontal street
        {"name": "LongStreet", "aliases": [],
         "geometry": "LINESTRING (8.530 47.370, 8.547 47.370)",
         "lat": 47.370, "lon": 8.5385},      # centroid (mid-point)
        # a small point POI far away from the query (but near centroid)
        {"name": "NearCentroid", "aliases": [],
         "geometry": "POINT (8.5385 47.3705)",
         "lat": 47.3705, "lon": 8.5385},
    ]
    idx = sp.build_poi_index(pois)
    # query point: on the street at its western end, far from centroid
    p, d = sp.nearest_poi_m(47.370, 8.531, idx)
    assert p["name"] == "LongStreet"          # point-to-LINE wins
    assert d < 50.0                            # right on the street


def test_empty_index_returns_none():
    idx = sp.build_poi_index([])
    p, d = sp.nearest_poi_m(47.37, 8.54, idx)
    assert p is None and d == float("inf")


def test_distance_pois_m_uses_geometries_not_centroids():
    """Two parallel streets: their centroids are far apart but the
    streets pass closely. distance_pois_m must return the *shortest*
    point-to-point distance between their geometries."""
    pois = [
        # two horizontal "streets" ~100 m apart (at lat 47.37 vs 47.371)
        {"name": "StreetA", "aliases": [],
         "geometry": "LINESTRING (8.500 47.370, 8.510 47.370)",
         "lat": 47.370, "lon": 8.505},
        {"name": "StreetB", "aliases": [],
         "geometry": "LINESTRING (8.500 47.371, 8.510 47.371)",
         "lat": 47.371, "lon": 8.505},
    ]
    idx = sp.build_poi_index(pois)
    d = sp.distance_pois_m("StreetA", "StreetB", idx)
    assert 100 < d < 115            # roughly 111 m (1/1000 of a degree lat)
    # missing names -> infinity
    assert sp.distance_pois_m("StreetA", "Nope", idx) == float("inf")
    assert sp.distance_pois_m("Nope", "StreetB", idx) == float("inf")


def test_name_match_exact_and_alias_and_diacritic():
    poi = {"name": "Grossmünster",
           "aliases": ["Great Minster Church"]}
    assert sp.name_matches_poi("Grossmünster", poi)
    assert sp.name_matches_poi("grossmunster", poi)          # diacritic
    assert sp.name_matches_poi("Great Minster Church", poi)  # alias
    assert sp.name_matches_poi("Bahnhofstrasse", poi) is False
    assert sp.name_matches_poi("", poi) is False
    assert sp.name_matches_poi("anything", None) is False
