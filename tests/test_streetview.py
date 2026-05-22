"""Unit tests for src/streetview.py — the pure bbox / grid geometry.

Network steps (scan, download) are not exercised here.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import streetview as sv  # noqa: E402
from src import poi               # noqa: E402


def test_bbox_from_pois_contains_every_poi():
    w, s, e, n = sv.bbox_from_pois(poi.CANDIDATE_POIS)
    for _, _, lat, lon, _ in poi.CANDIDATE_POIS:
        assert w < lon < e
        assert s < lat < n


def test_bbox_has_outward_margin():
    pts = [(47.37, 8.54), (47.38, 8.55)]   # (lat, lon) pairs
    w, s, e, n = sv.bbox_from_pois(pts, margin_m=300)
    assert w < 8.54 and s < 47.37          # extends below/left of the min
    assert e > 8.55 and n > 47.38          # and above/right of the max


def test_grid_points_lie_inside_bbox():
    bbox = (8.52, 47.36, 8.54, 47.38)
    pts = sv.grid_points(bbox, spacing_m=100)
    assert len(pts) > 4
    for lat, lon in pts:
        assert bbox[1] <= lat <= bbox[3]
        assert bbox[0] <= lon <= bbox[2]


def test_grid_denser_with_smaller_spacing():
    bbox = (8.52, 47.36, 8.55, 47.39)
    assert len(sv.grid_points(bbox, 30)) > len(sv.grid_points(bbox, 100))


def test_crawl_bbox_well_formed():
    w, s, e, n = sv.crawl_bbox()
    assert w < e and s < n


def test_bbox_from_scan(tmp_path):
    pois = tmp_path / "pois.json"
    scan = tmp_path / "poi_scan.jsonl"
    pois.write_text(json.dumps([
        {"name": "A", "lat": 47.37, "lon": 8.54},
        {"name": "B", "lat": 47.38, "lon": 8.55},
        {"name": "C", "lat": 47.36, "lon": 8.53}]), encoding="utf-8")
    scan.write_text(
        json.dumps({"matched": [{"osm_name": "A"}, {"osm_name": "B"}]}) + "\n"
        + json.dumps({"matched": [{"osm_name": "C"}]}) + "\n", encoding="utf-8")
    w, s, e, n = sv.bbox_from_scan(scan, pois, margin_m=300)
    # bbox of A/B/C (lat 47.36-47.38, lon 8.53-8.55) + a 300 m margin
    assert w < 8.53 and s < 47.36 and e > 8.55 and n > 47.38
