"""Complete OSM shortest-path grid among the top-N POI destinations.

For each of the **top-N POIs** (default 30, the destination pool the
teacher annotator draws from in §2.7), this draws the OSM walking-
graph shortest path between every unordered pair. With N=30 that's
C(30, 2) = **435 routes** plotted as semi-transparent thin lines on
one Leaflet map. Where many routes overlap (Bahnhofstrasse, Limmat-
quai, the main old-town axes) the visual density compounds into a
natural heat-map of "which streets the trained model will most rely
on for routing".

Each POI's representative location is the **median GPS of all frames
in `gps_recovery_full.jsonl` whose `place_guess` resolves to that
POI** — not the OSM-table centroid (which for long streets like
Bahnhofstrasse is far from where the videos actually walk). This
matches the destination GPS the annotator will use.

Output: `viz/poi_route_grid.html` — one Leaflet map with:
  - 30 markers, labelled, coloured by visit-count rank
  - all C(30,2) routes overlaid (toggleable as one layer)
  - POI markers and routes are separate layers (toggle independently)

  python -m src.viz_poi_route_grid
  python -m src.viz_poi_route_grid --top-n 50          # widen the pool
  python -m src.viz_poi_route_grid --max-route-km 1.5  # cap edge length
"""

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402


def _osm_path_latlon(G, start_latlon, end_latlon,
                     to_latlon_xform=None, to_proj_xform=None,
                     is_projected=False):
    """Shortest walking path between two (lat, lon) points on G.
    Returns [(lat, lon), ...] or empty list if unreachable."""
    import networkx as nx
    import osmnx as ox

    def _nearest(lat, lon):
        if is_projected:
            x, y = to_proj_xform.transform(lon, lat)
            return ox.distance.nearest_nodes(G, x, y)
        return ox.distance.nearest_nodes(G, lon, lat)

    def _node_latlon(n):
        attr = G.nodes[n]
        if is_projected:
            lon, lat = to_latlon_xform.transform(attr["x"], attr["y"])
            return (lat, lon)
        return (attr["y"], attr["x"])

    a = _nearest(*start_latlon)
    b = _nearest(*end_latlon)
    if a == b:
        return [_node_latlon(a)]
    try:
        nodes = nx.shortest_path(G, a, b, weight="length")
    except nx.NetworkXNoPath:
        return []
    return [_node_latlon(n) for n in nodes]


def _path_length_m(latlon_path):
    """Total length of a polyline in metres (haversine sum)."""
    import math
    R = 6_371_000.0
    if len(latlon_path) < 2:
        return 0.0
    out = 0.0
    for (la1, lo1), (la2, lo2) in zip(latlon_path, latlon_path[1:]):
        p1, p2 = math.radians(la1), math.radians(la2)
        dphi = math.radians(la2 - la1)
        dlam = math.radians(lo2 - lo1)
        a = (math.sin(dphi / 2) ** 2
             + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2)
        out += 2 * R * math.asin(math.sqrt(a))
    return out


def top_poi_centroids(rows, n, field="place_guess"):
    """Top-n POI names by frame count + each one's median (lat, lon)
    across frames whose `field` resolves to that POI."""
    counts = collections.Counter()
    points = collections.defaultdict(list)
    for r in rows:
        name = r.get(field) or ""
        if not name:
            continue
        counts[name] += 1
        if r.get("gps"):
            points[name].append(tuple(r["gps"]))
    common = [name for name, _ in counts.most_common(n)]
    out = []
    for name in common:
        pts = points.get(name, [])
        if not pts:
            continue
        lats = sorted(p[0] for p in pts)
        lons = sorted(p[1] for p in pts)
        # median, robust to outliers
        out.append({
            "name": name,
            "count": counts[name],
            "lat": lats[len(lats) // 2],
            "lon": lons[len(lons) // 2],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "gps_recovery_full.jsonl"),
                    help="frames jsonl that carries place_guess + gps "
                         "(default: gps_recovery_full.jsonl, "
                         "VLM-agreed accepted rows used to compute the "
                         "top-N pool)")
    ap.add_argument("--graph",
                    default=str(config.CITY_DIR / "osm_walking.pkl"))
    ap.add_argument("--output",
                    default=str(config.VIZ_DIR / "poi_route_grid.html"))
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--poi-field", default="place_guess")
    ap.add_argument("--tier", type=int, choices=[0, 1, 2], default=1,
                    help="filter input rows by tier (default 1 = "
                         "VLM-agreed only)")
    ap.add_argument("--max-route-km", type=float, default=0.0,
                    help="skip routes longer than this (0 = no cap)")
    args = ap.parse_args()

    def _resolve(p):
        path = Path(p)
        if path.exists() or path.is_absolute():
            return path
        in_city = config.CITY_DIR / path.name
        return in_city if in_city.exists() else path

    in_path = _resolve(args.input)
    graph_path = _resolve(args.graph)
    if not in_path.exists():
        sys.exit(f"[viz_poi_route_grid] input not found: {in_path}")
    if not graph_path.exists():
        sys.exit(f"[viz_poi_route_grid] osm graph not found: {graph_path}\n"
                 f"  build it: python -m src.build_walking_graph")

    rows = []
    for line in in_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        # `accepted` may be missing (trusted_frames.jsonl is already
        # filtered to accepted-only by construction). Treat missing
        # as pass.
        if r.get("accepted") is False:
            continue
        if args.tier and r.get("tier") not in (None, args.tier):
            continue
        rows.append(r)
    print(f"[viz_poi_route_grid] in: {in_path.name}  "
          f"(tier={args.tier} accepted: {len(rows):,} frames)",
          flush=True)

    pois = top_poi_centroids(rows, args.top_n, field=args.poi_field)
    print(f"[viz_poi_route_grid] top-{args.top_n} POIs ({args.poi_field}):",
          flush=True)
    for p in pois:
        print(f"   {p['name']:30s}  {p['count']:4d} frames  "
              f"({p['lat']:.5f}, {p['lon']:.5f})", flush=True)

    import pickle
    with graph_path.open("rb") as f:
        G = pickle.load(f)
    crs = G.graph.get("crs")
    is_projected = bool(crs) and "4326" not in str(crs)
    to_latlon = to_proj = None
    if is_projected:
        from pyproj import Transformer
        to_latlon = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Compute ALL C(N, 2) shortest paths once. Tiny cohort (N=30)
    # so we don't bother caching node-pair lookups across calls.
    print(f"[viz_poi_route_grid] computing shortest paths …", flush=True)
    edges = []
    skipped_unreachable = skipped_long = 0
    pairs = list(itertools.combinations(range(len(pois)), 2))
    for n_done, (i, j) in enumerate(pairs, 1):
        a, b = pois[i], pois[j]
        path = _osm_path_latlon(G, (a["lat"], a["lon"]),
                                 (b["lat"], b["lon"]),
                                 to_latlon, to_proj, is_projected)
        if not path:
            skipped_unreachable += 1
            continue
        length_m = _path_length_m(path)
        if args.max_route_km and length_m / 1000.0 > args.max_route_km:
            skipped_long += 1
            continue
        edges.append({"a": a["name"], "b": b["name"],
                      "path": path, "length_m": length_m})
        if n_done % 50 == 0:
            print(f"   {n_done}/{len(pairs)} pairs done", flush=True)
    print(f"[viz_poi_route_grid] {len(edges)} routes plotted  "
          f"(skipped {skipped_unreachable} unreachable + "
          f"{skipped_long} over {args.max_route_km} km)",
          flush=True)

    # ─── render ────────────────────────────────────────────────────
    import folium
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    W, S, E, N = config.POI_BBOX
    m = folium.Map(location=[(S + N) / 2, (W + E) / 2],
                   zoom_start=15, tiles="cartodbpositron")
    folium.Rectangle(bounds=[[S, W], [N, E]],
                     color="#000", weight=1, fill=False).add_to(m)

    # All routes as one toggleable layer — semi-transparent so
    # overlap density acts as a heatmap of "which streets are the
    # main corridors".
    routes_fg = folium.FeatureGroup(
        name=f"all {len(edges)} OSM routes (overlay heatmap)", show=True)
    # Color routes by length (short blue → long red) so user can tell
    # corner-to-corner from across-town
    if edges:
        lens = [e["length_m"] for e in edges]
        lmin, lmax = min(lens), max(lens)
        cmap = cm.get_cmap("viridis_r")     # short = yellow, long = purple
    for e in edges:
        frac = ((e["length_m"] - lmin) / max(1.0, lmax - lmin)
                if edges else 0.5)
        color = mcolors.to_hex(cmap(frac))
        folium.PolyLine(
            e["path"], color=color, weight=2, opacity=0.18,
            tooltip=f"{e['a']} → {e['b']}  ({e['length_m']:.0f} m)",
        ).add_to(routes_fg)
    routes_fg.add_to(m)

    # POI markers — coloured by visit-count rank (top 1 red, fading
    # to grey for the lowest of the top-N).
    pois_fg = folium.FeatureGroup(
        name=f"top-{len(pois)} POIs (centroid of VLM-agreed frames)",
        show=True)
    counts = [p["count"] for p in pois]
    cmax, cmin = max(counts), min(counts)
    rank_cmap = cm.get_cmap("Reds")
    for rank, p in enumerate(pois, 1):
        frac = (p["count"] - cmin) / max(1.0, cmax - cmin)
        marker_color = mcolors.to_hex(rank_cmap(0.35 + 0.55 * frac))
        folium.CircleMarker(
            [p["lat"], p["lon"]], radius=8, color="#000",
            weight=1, fill=True, fill_color=marker_color,
            fill_opacity=0.95,
            tooltip=f"#{rank}  {p['name']}  ({p['count']} frames)",
            popup=folium.Popup(
                f"<b>#{rank} {p['name']}</b><br>"
                f"{p['count']} frames in cohort<br>"
                f"GPS {p['lat']:.5f}, {p['lon']:.5f}", max_width=240),
        ).add_to(pois_fg)
        # numeric label next to each dot
        folium.map.Marker(
            [p["lat"], p["lon"]],
            icon=folium.DivIcon(
                icon_size=(120, 18), icon_anchor=(-10, 8),
                html=(f'<div style="font:11px/1.1 system-ui;'
                      f'color:#222;background:rgba(255,255,255,0.85);'
                      f'padding:1px 4px;border:1px solid #888;'
                      f'border-radius:3px;display:inline-block;">'
                      f'{rank}. {p["name"]}</div>')),
        ).add_to(pois_fg)
    pois_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    title_html = (
        '<div style="position:fixed;top:10px;left:50px;z-index:9999;'
        'background:white;padding:8px 12px;border:1px solid #888;'
        'font:13px/1.4 system-ui;max-width:340px">'
        f'<b>Top-{len(pois)} POI route grid</b><br>'
        f'{len(edges)} OSM shortest paths overlaid<br>'
        f'<i>Route colour</i>: viridis_r — yellow = short, '
        f'purple = long<br>'
        f'<i>Marker fill</i>: rank by VLM-agreed-frame count<br>'
        f'Toggle the routes/markers layers top-right.'
        '</div>')
    m.get_root().html.add_child(folium.Element(title_html))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"[viz_poi_route_grid] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
