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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402

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


def build_map(per_video, G, is_projected, to_latlon, to_proj):
    import folium

    W, S, E, N = config.POI_BBOX
    m = folium.Map(location=[(S + N) / 2, (W + E) / 2], zoom_start=15,
                   tiles="cartodbpositron")
    folium.Rectangle(bounds=[[S, W], [N, E]],
                     color="#000", weight=1, fill=False).add_to(m)

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
        osm_path = _osm_path_latlon(G, start, mid, end,
                                     to_latlon, to_proj, is_projected)
        if osm_path:
            folium.PolyLine(osm_path, color="#888", weight=6,
                            opacity=0.55, dash_array="8,8",
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
        '<span style="color:#888">━ ━ ━</span> OSM ideal '
        '(shortest path start→mid→end)<br>'
        '<span style="color:#444">━━━</span> recovered (actual walk) — '
        '<i>dark→light = start→end</i><br>'
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

    m = build_map(per_video, G, is_projected, to_latlon, to_proj)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"[viz_route_compare] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
