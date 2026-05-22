"""Unit tests for src/viz.py — the Leaflet HTML builders."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import viz  # noqa: E402


def test_leaflet_page_embeds_coords_and_colour():
    html = viz.leaflet_page("T", (47.37, 8.54), [
        {"coords": [[47.37, 8.54], [47.38, 8.55]],
         "color": "#f00", "label": "x"}])
    assert "leaflet" in html.lower()
    assert "47.38" in html and "8.55" in html
    assert "#f00" in html


def test_route_map_writes_html(tmp_path):
    out = tmp_path / "rm.html"
    routes = {"v1": [(47.37, 8.54), (47.38, 8.55)],
              "v2": [(47.36, 8.53), (47.37, 8.54)]}
    p = viz.route_map(routes, out=out)
    assert p.exists()
    assert "polyline" in p.read_text(encoding="utf-8")


def test_eight_distinct_colours():
    assert len(viz.COLORS) == 8
    assert len(set(viz.COLORS)) == 8
