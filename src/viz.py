"""Visualizations — standalone Leaflet HTML (DEV_MANUAL §5).

`leaflet_page()` is a pure HTML builder (unit-tested). `route_map()`
draws the 8-video recovered-GPS routes as coloured polylines. The
27-candidate POI map lives in `src/poi.py --map`.

    python -m src.viz --routes <routes.json>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# 8 distinct colours — one per source video
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
          "#911eb4", "#1abc9c", "#f032e6", "#808000"]


def leaflet_page(title, center, polylines):
    """Build a standalone Leaflet HTML page. Pure function.

    polylines: list of {coords: [[lat,lon],...], color: str, label: str}.
    Returns the HTML string.
    """
    clat, clon = center
    lines = []
    for pl in polylines:
        coords = json.dumps(pl["coords"])
        lines.append(
            f"L.polyline({coords},{{color:'{pl['color']}',weight:4,"
            f"opacity:.8}}).bindPopup('{pl.get('label','')}').addTo(map);")
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>"
        "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>"
        "<style>html,body,#map{height:100%;margin:0}</style></head><body>"
        "<div id='map'></div><script>"
        f"var map=L.map('map').setView([{clat},{clon}],14);"
        "L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',"
        "{maxZoom:19,attribution:'© OpenStreetMap'}).addTo(map);"
        + "".join(lines) +
        "</script></body></html>"
    )


def route_map(routes, out=None):
    """routes: {video_name: [(lat,lon), ...]}. Writes an HTML route map.

    Returns the output Path.
    """
    polylines, all_lat, all_lon = [], [], []
    for i, (name, pts) in enumerate(sorted(routes.items())):
        if not pts:
            continue
        polylines.append({"coords": [[la, lo] for la, lo in pts],
                          "color": COLORS[i % len(COLORS)], "label": name})
        all_lat += [p[0] for p in pts]
        all_lon += [p[1] for p in pts]
    center = ((sum(all_lat) / len(all_lat), sum(all_lon) / len(all_lon))
              if all_lat else (47.37, 8.54))
    html = leaflet_page("NavLM — 8-video routes", center, polylines)

    out = out or (config.VIZ_DIR / "route_map.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"route map ({len(polylines)} routes) -> {out}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="visualizations")
    ap.add_argument("--routes", help="JSON {video: [[lat,lon],...]}")
    args = ap.parse_args()
    if args.routes:
        route_map({k: [tuple(p) for p in v]
                   for k, v in json.loads(Path(args.routes).read_text()).items()})
    else:
        print("pass --routes <routes.json>; the POI map is `src.poi --map`.")


if __name__ == "__main__":
    main()
