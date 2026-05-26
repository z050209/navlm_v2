"""Per-video route comparison: actual recovered walk vs OSM shortest path.

For each video in the input, draws TWO polylines on the same map:

  RECOVERED route — the chronological sequence of road_snapped GPS
                    positions (the path the videographer actually
                    walked, as our pipeline reconstructs it).
  OSM IDEAL route — the OSM walking-graph shortest path from the
                    first frame's GPS to the last frame's GPS,
                    passing through the middle frame as a waypoint
                    so loops still produce a meaningful comparison.

Output is ONE Leaflet/folium HTML with per-video toggleable layer
groups (start with all on; click the layer control top-right to focus
one video at a time).

Markers:
  green pin   = first frame (start of the walk)
  red pin     = last frame  (end)
  blue pin    = middle waypoint used by the OSM shortest path

Comparing the two polylines per video answers:
  - does our pipeline's recovered route make geographic sense?
  - did the videographer take the shortest possible path, or
    meander through landmarks (the tour-walk pattern)?
  - are there segments where the recovered route jumps far off the
    OSM graph (sign of a remaining gps_recovery error)?

  python -m src.viz_route_compare                     # default inputs
  python -m src.viz_route_compare --input trusted_frames.jsonl
  python -m src.viz_route_compare --only saturday_morning
"""

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402
from src.viz_poi_route_grid import (           # noqa: E402
    _osm_path_latlon as _osm_path_pair,
    top_poi_centroids,
)

VIDEO_COLORS = {
    "bahnhofstrasse":   "#e6194B",
    "hidden_streets":   "#3cb44b",
    "looks_perfect":    "#cc9e00",       # darker yellow base so the
                                          # gradient stays visible on
                                          # the cartodbpositron tile
    "most_elegant":     "#4363d8",
    "most_famous":      "#f58231",
    "old_town_limmat":  "#911eb4",
    "zurich_main":      "#1899b8",
    "saturday_morning": "#000000",       # eval hold-out
}
GRADIENT_CHUNKS = 20                      # segments along each video's
                                          # recovered line — dark = start,
                                          # light = end (= direction cue)


def _shade(hex_base, frac):
    """Interpolate the base colour from a dark variant (frac=0, ~40 %
    luminance) to a light variant (frac=1, ~halfway to white). Pure."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.hex2color(hex_base)
    dark = (r * 0.35, g * 0.35, b * 0.35)
    light = (r + (1.0 - r) * 0.55,
             g + (1.0 - g) * 0.55,
             b + (1.0 - b) * 0.55)
    out = tuple(dark[i] + (light[i] - dark[i]) * frac for i in range(3))
    return mcolors.to_hex(out)


def _gradient_segments(coords, n_chunks=GRADIENT_CHUNKS):
    """Split `coords` (list of [lat, lon]) into `n_chunks` consecutive
    segments. Each segment is a sub-list of coords; consecutive
    segments share a vertex so the line is continuous. Returns a list
    of (segment_coords, frac) where frac in [0, 1] maps start→end."""
    n = len(coords)
    if n <= 1:
        return []
    chunks = min(n_chunks, n - 1)
    edges_per_chunk = max(1, (n - 1) // chunks)
    out = []
    i = 0
    chunk_idx = 0
    while i < n - 1 and chunk_idx < chunks:
        j = min(n - 1,
                (i + edges_per_chunk if chunk_idx < chunks - 1 else n - 1))
        sub = coords[i:j + 1]
        frac = (chunk_idx / (chunks - 1)) if chunks > 1 else 1.0
        out.append((sub, frac))
        i = j
        chunk_idx += 1
    return out


def _osm_path_latlon(G, start_latlon, mid_latlon, end_latlon,
                     to_latlon_xform=None, to_proj_xform=None,
                     is_projected=False):
    """Return [(lat, lon)] of the OSM shortest walking path
    start -> mid -> end. Empty list if unreachable."""
    import networkx as nx
    import osmnx as ox

    def _nearest(lat, lon):
        if is_projected:
            x, y = to_proj_xform.transform(lon, lat)
            return ox.distance.nearest_nodes(G, x, y)
        return ox.distance.nearest_nodes(G, lon, lat)

    def _node_latlon(node):
        n = G.nodes[node]
        if is_projected:
            lon, lat = to_latlon_xform.transform(n["x"], n["y"])
            return (lat, lon)
        return (n["y"], n["x"])

    sn = _nearest(*start_latlon)
    mn = _nearest(*mid_latlon)
    en = _nearest(*end_latlon)
    try:
        leg1 = nx.shortest_path(G, sn, mn, weight="length")
        leg2 = nx.shortest_path(G, mn, en, weight="length")
    except nx.NetworkXNoPath:
        return []
    nodes = leg1 + leg2[1:]                  # don't repeat mn
    return [_node_latlon(n) for n in nodes]


def _add_poi_grid_layers(m, poi_rows, G, is_projected, to_latlon,
                          to_proj, top_n=30):
    """Add two layers to the map: (a) top-N POI markers labelled by
    rank + name (default ON), (b) all C(N,2) OSM shortest paths
    between the POIs as a faded background heatmap (default OFF, so
    the per-video lines stay readable when the user opens the page)."""
    import folium
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    pois = top_poi_centroids(poi_rows, top_n, field="place_guess")
    if not pois:
        print("[viz_route_compare] no POI rows available for the grid "
              "layer (skipped)", flush=True)
        return

    # ─── 435-route background (off by default — toggle to enable)
    routes_fg = folium.FeatureGroup(
        name=f"top-{len(pois)} POI route grid ({len(pois)*(len(pois)-1)//2} "
             f"OSM paths)  [off by default]",
        show=False)
    pairs = list(itertools.combinations(range(len(pois)), 2))
    print(f"[viz_route_compare] computing POI-pair grid "
          f"({len(pairs)} paths) …", flush=True)
    lengths = []
    paths = []
    for i, j in pairs:
        a, b = pois[i], pois[j]
        path = _osm_path_pair(G, (a["lat"], a["lon"]),
                               (b["lat"], b["lon"]),
                               to_latlon, to_proj, is_projected)
        if not path or len(path) < 2:
            continue
        # rough length for colouring (sum of haversine segments — close
        # enough for visual sorting)
        import math
        R = 6_371_000.0
        L = 0.0
        for (la1, lo1), (la2, lo2) in zip(path, path[1:]):
            p1, p2 = math.radians(la1), math.radians(la2)
            dphi = math.radians(la2 - la1)
            dlam = math.radians(lo2 - lo1)
            a_ = (math.sin(dphi / 2) ** 2
                  + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2)
            L += 2 * R * math.asin(math.sqrt(a_))
        lengths.append(L)
        paths.append((pois[i]["name"], pois[j]["name"], path, L))
    if paths:
        lmin, lmax = min(lengths), max(lengths)
        cmap = cm.get_cmap("viridis_r")
        for name_a, name_b, path, L in paths:
            frac = (L - lmin) / max(1.0, lmax - lmin)
            folium.PolyLine(
                path, color=mcolors.to_hex(cmap(frac)),
                weight=2, opacity=0.15,
                tooltip=f"{name_a} → {name_b}  ({L:.0f} m)",
            ).add_to(routes_fg)
    routes_fg.add_to(m)
    print(f"[viz_route_compare] POI grid: {len(paths)} paths added "
          f"(toggle on in the layer control)", flush=True)

    # ─── POI markers (default ON — useful context for every video)
    pois_fg = folium.FeatureGroup(
        name=f"top-{len(pois)} POIs (destination pool)", show=True)
    counts = [p["count"] for p in pois]
    cmax, cmin = max(counts), min(counts)
    rank_cmap = cm.get_cmap("Reds")
    for rank, p in enumerate(pois, 1):
        frac = (p["count"] - cmin) / max(1.0, cmax - cmin)
        fill = mcolors.to_hex(rank_cmap(0.35 + 0.55 * frac))
        folium.CircleMarker(
            [p["lat"], p["lon"]], radius=7, color="#000",
            weight=1, fill=True, fill_color=fill, fill_opacity=0.95,
            tooltip=f"#{rank}  {p['name']}  ({p['count']} frames)",
        ).add_to(pois_fg)
        folium.map.Marker(
            [p["lat"], p["lon"]],
            icon=folium.DivIcon(
                icon_size=(140, 18), icon_anchor=(-8, 8),
                html=(f'<div style="font:11px/1.1 system-ui;'
                      f'color:#222;background:rgba(255,255,255,0.88);'
                      f'padding:1px 4px;border:1px solid #888;'
                      f'border-radius:3px;display:inline-block;">'
                      f'{rank}. {p["name"]}</div>')),
        ).add_to(pois_fg)
    pois_fg.add_to(m)


def build_map(per_video, G, is_projected, to_latlon, to_proj,
              poi_rows=None, top_n=30):
    import folium

    W, S, E, N = config.POI_BBOX
    m = folium.Map(location=[(S + N) / 2, (W + E) / 2], zoom_start=15,
                   tiles="cartodbpositron")
    folium.Rectangle(bounds=[[S, W], [N, E]],
                     color="#000", weight=1, fill=False).add_to(m)

    # POI grid layers — added FIRST so they sit beneath the per-video
    # polylines in z-order (the routes layer is off by default; the
    # POI markers are on so each per-video map has destination
    # context).
    if poi_rows:
        _add_poi_grid_layers(m, poi_rows, G, is_projected, to_latlon,
                              to_proj, top_n=top_n)

    summary = []
    for video, rows in sorted(per_video.items()):
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: r["frame_id"])
        coords = [r["gps"] for r in rows]
        start = coords[0]
        end = coords[-1]
        mid = coords[len(coords) // 2]
        color = VIDEO_COLORS.get(video, "#888")

        fg = folium.FeatureGroup(name=f"{video} ({len(rows)})", show=True)

        # OSM IDEAL — drawn first so it's behind the recovered line.
        # Coloured in the VIDEO'S OWN hue (lighter shade + dashed +
        # reduced opacity) so 8 OSM routes overlapping in central
        # Zurich stay visually distinguishable per video. Previously
        # they were all grey #888 and collapsed into a single
        # indistinguishable mass.
        osm_path = _osm_path_latlon(G, start, mid, end,
                                     to_latlon, to_proj, is_projected)
        if osm_path:
            osm_color = _shade(color, 1.0)        # light variant
            folium.PolyLine(osm_path, color=osm_color, weight=5,
                            opacity=0.85, dash_array="10,6",
                            tooltip=f"OSM ideal · {video}").add_to(fg)

        # RECOVERED — the actual walked path, on top. Drawn as
        # GRADIENT_CHUNKS segments shaded dark→light so the direction
        # of travel is visible without arrows.
        for sub, frac in _gradient_segments(coords):
            folium.PolyLine(sub, color=_shade(color, frac), weight=5,
                            opacity=0.95,
                            tooltip=(f"recovered · {video} "
                                     f"· {int(100*frac):>3d} % through "
                                     f"({len(coords)} frames)")
                            ).add_to(fg)

        # frame dots (small) so the user can click any to inspect
        for r in rows:
            folium.CircleMarker(
                r["gps"], radius=2, color=color, fill=True,
                fill_color=color, fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>{video}</b><br>{r['frame_id']}<br>"
                    f"heading {r.get('heading') or 0:.0f}&deg;",
                    max_width=220),
            ).add_to(fg)

        # markers
        folium.Marker(start, icon=folium.Icon(color="green", icon="play"),
                      tooltip=f"start · {video}/{rows[0]['frame_id']}"
                      ).add_to(fg)
        folium.Marker(end, icon=folium.Icon(color="red", icon="stop"),
                      tooltip=f"end · {video}/{rows[-1]['frame_id']}"
                      ).add_to(fg)
        folium.Marker(mid, icon=folium.Icon(color="blue", icon="record"),
                      tooltip=f"OSM waypoint · {video}/"
                              f"{rows[len(rows)//2]['frame_id']}"
                      ).add_to(fg)

        fg.add_to(m)
        summary.append((video, len(rows), len(osm_path) if osm_path else 0))

    folium.LayerControl(collapsed=False).add_to(m)

    legend_chips = []
    for v, n, o in summary:
        base = VIDEO_COLORS.get(v, "#888")
        dark = _shade(base, 0.0)
        light = _shade(base, 1.0)
        legend_chips.append(
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"background:{dark};margin-right:0px'></span>"
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"background:{base};margin-right:0px'></span>"
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"background:{light};margin-right:4px'></span>"
            f"{v} — {n} frames, OSM {o} nodes")
    title_html = (
        '<div style="position:fixed;top:10px;left:50px;z-index:9999;'
        'background:white;padding:8px 12px;border:1px solid #888;'
        'font:13px/1.4 system-ui;max-width:380px">'
        '<b>Per-video route comparison</b><br>'
        '<span style="color:#888">━ ━ ━</span> OSM ideal per video '
        '(shortest start→mid→end, video hue)<br>'
        '<span style="color:#444">━━━</span> recovered walk '
        '— <i>dark→light = start→end</i><br>'
        '<span style="color:#c0392b">●</span> top-30 POI destination '
        'pool (always-on layer)<br>'
        '<span style="color:#888">grid</span> 435 POI-pair OSM routes '
        '(toggle <i>OFF by default</i>)<br>'
        '<br>'
        + "<br>".join(legend_chips)
        + '</div>')
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "road_snapped.jsonl"),
                    help="per-frame jsonl carrying {video, frame_id, gps} "
                         "(road_snapped.jsonl preferred; "
                         "trusted_frames.jsonl also works)")
    ap.add_argument("--graph",
                    default=str(config.CITY_DIR / "osm_walking.pkl"))
    ap.add_argument("--output",
                    default=str(config.VIZ_DIR /
                                "route_compare_per_video.html"))
    ap.add_argument("--only", default="",
                    help="render only one video (dataset name)")
    ap.add_argument("--poi-input",
                    default=str(config.CITY_DIR / "trusted_frames.jsonl"),
                    help="jsonl whose place_guess column ranks the top-N "
                         "destination POIs to overlay (default: "
                         "trusted_frames.jsonl — same 30 POIs the "
                         "annotator will use). Pass --no-poi-grid to skip.")
    ap.add_argument("--poi-top-n", type=int, default=30)
    ap.add_argument("--no-poi-grid", action="store_true",
                    help="skip the POI grid layers entirely")
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
        sys.exit(f"[viz_route_compare] input not found: {in_path}")
    if not graph_path.exists():
        sys.exit(f"[viz_route_compare] OSM graph not found: {graph_path}\n"
                 f"  build it: python -m src.build_walking_graph")

    rows = []
    for line in in_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        # road_snapped uses gps_snapped; trusted uses gps. Normalize:
        gps = r.get("gps_snapped") or r.get("gps")
        if gps is None:
            continue
        rows.append({"video": r["video"], "frame_id": r["frame_id"],
                     "gps": gps, "heading": r.get("heading")})
    if args.only:
        rows = [r for r in rows if r["video"] == args.only]
    per_video = collections.defaultdict(list)
    for r in rows:
        per_video[r["video"]].append(r)

    print(f"[viz_route_compare] in: {in_path.name}  "
          f"({sum(len(v) for v in per_video.values())} frames, "
          f"{len(per_video)} videos)", flush=True)

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
        print(f"[viz_route_compare] graph CRS: {crs}", flush=True)

    # Load POI-cohort rows (separate from per_video — typically the
    # trusted_frames.jsonl with `place_guess` and `gps`).
    poi_rows = []
    if not args.no_poi_grid:
        poi_path = _resolve(args.poi_input)
        if poi_path.exists():
            for line in poi_path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("accepted") is False:
                    continue
                if r.get("gps") is None:
                    continue
                poi_rows.append(r)
            print(f"[viz_route_compare] POI grid source: {poi_path.name} "
                  f"({len(poi_rows)} rows)", flush=True)
        else:
            print(f"[viz_route_compare] POI grid source not found: "
                  f"{poi_path} — skipping POI grid", flush=True)

    m = build_map(per_video, G, is_projected, to_latlon, to_proj,
                  poi_rows=poi_rows if poi_rows else None,
                  top_n=args.poi_top_n)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"[viz_route_compare] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
