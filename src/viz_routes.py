"""Per-video route map.

For every video, draws the **time-ordered** GPS sequence as a coloured
polyline on one Leaflet/folium map, with frame markers and arrows
showing the camera heading at each frame. The point of this map is to
*eyeball* whether GPS recovery is tracing the right path, before we
sink Pro-dollars into instruction annotation.

  python -m src.viz_routes                       # all videos, default input
  python -m src.viz_routes --input phaseA_trusted.jsonl
  python -m src.viz_routes --only saturday_morning --every-n 5
  python -m src.viz_routes --show-headings       # add a heading arrow

Input: any per-frame jsonl with `{video, frame_id, gps:[lat,lon],
heading?}` — works on `gps_recovery_all.jsonl` (raw), `phaseA_snapped.jsonl`
(HMM-snapped) or `phaseA_trusted.jsonl` (after heading_qc). Output:
`viz/routes_<input_stem>.html`.

`frame_id` is sortable lexicographically (`frame_00000`, `frame_00001`,
…) so plain sort gives the timeline.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402

# Distinct colours for 8 videos — keep saturday_morning (hold-out)
# black so it's instantly distinguishable on the map.
VIDEO_COLORS = {
    "bahnhofstrasse":   "#e6194B",
    "hidden_streets":   "#3cb44b",
    "looks_perfect":    "#ffe119",
    "most_elegant":     "#4363d8",
    "most_famous":      "#f58231",
    "old_town_limmat":  "#911eb4",
    "zurich_main":      "#42d4f4",
    "saturday_morning": "#000000",   # hold-out
}


def _heading_endpoint(lat, lon, heading_deg, length_m=8.0):
    """Tiny endpoint a few metres along `heading_deg` for an arrow."""
    R = 6_371_000.0
    br = math.radians(heading_deg)
    dlat = (length_m * math.cos(br)) / R
    dlon = (length_m * math.sin(br)) / (R * math.cos(math.radians(lat)))
    return (lat + math.degrees(dlat), lon + math.degrees(dlon))


def build_map(rows_by_video, show_headings=False, every_n=1):
    import folium

    # centre on the POI bbox so an empty map still renders sensibly
    W, S, E, N = config.POI_BBOX
    m = folium.Map(location=[(S + N) / 2, (W + E) / 2], zoom_start=14,
                   tiles="cartodbpositron")

    folium.Rectangle(bounds=[[S, W], [N, E]],
                     color="#000", weight=1, fill=False,
                     tooltip="POI_BBOX").add_to(m)

    counts = {}
    for video, rows in sorted(rows_by_video.items()):
        rows = sorted(rows, key=lambda r: r["frame_id"])
        colour = VIDEO_COLORS.get(video, "#888888")
        fg = folium.FeatureGroup(name=f"{video} ({len(rows)})", show=True)

        coords = [r["gps"] for r in rows]
        if len(coords) >= 2:
            folium.PolyLine(coords, color=colour, weight=3,
                            opacity=0.7).add_to(fg)

        for i, r in enumerate(rows):
            if i % every_n != 0:
                continue
            lat, lon = r["gps"]
            folium.CircleMarker(
                [lat, lon], radius=2, color=colour, fill=True,
                fill_color=colour, fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>{r['video']}</b><br>{r['frame_id']}<br>"
                    f"heading {r.get('heading', '?'):.0f}°<br>"
                    f"gap {r.get('heading_gap', 0):.2f}",
                    max_width=240),
            ).add_to(fg)
            h = r.get("heading")
            if show_headings and h is not None:
                lat2, lon2 = _heading_endpoint(lat, lon, h, length_m=8.0)
                folium.PolyLine([[lat, lon], [lat2, lon2]],
                                color=colour, weight=2,
                                opacity=0.9).add_to(fg)

        counts[video] = len(rows)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    title_html = (
        '<div style="position:fixed;top:10px;left:50px;z-index:9999;'
        'background:white;padding:8px 12px;border:1px solid #888;'
        'font:13px/1.3 system-ui">'
        f'<b>NavLM routes — per video</b><br>' +
        "<br>".join(f"<span style='color:{VIDEO_COLORS.get(v, '#888')}'>■</span> "
                    f"{v} — {c} frames"
                    for v, c in sorted(counts.items())) +
        '</div>')
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "phaseA_trusted.jsonl"),
                    help="Per-frame jsonl with {video, frame_id, gps[,heading]}")
    ap.add_argument("--output", default=None,
                    help="Output HTML (default: viz/routes_<stem>.html)")
    ap.add_argument("--only", default=None,
                    help="Plot one video only (dataset name).")
    ap.add_argument("--every-n", type=int, default=1,
                    help="Down-sample frame markers (line stays full).")
    ap.add_argument("--show-headings", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        # graceful fallback: phaseA_trusted may not exist yet
        alt = config.CITY_DIR / "gps_recovery_all.jsonl"
        if alt.exists():
            print(f"[viz_routes] {in_path.name} not found, falling back "
                  f"to {alt.name}", flush=True)
            in_path = alt
        else:
            sys.exit(f"[viz_routes] no input found ({in_path}, {alt})")

    rows_by_video = defaultdict(list)
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if args.only and r["video"] != args.only:
                continue
            # gps_recovery_all uses {"gps": ..., "accepted": True}; trusted
            # has only kept rows (already filtered).
            if "gps" not in r:
                continue
            if r.get("accepted") is False:
                continue
            rows_by_video[r["video"]].append(r)

    print(f"[viz_routes] loaded {sum(len(v) for v in rows_by_video.values())}"
          f" rows from {in_path.name} "
          f"({len(rows_by_video)} video(s))", flush=True)

    m = build_map(rows_by_video, show_headings=args.show_headings,
                  every_n=args.every_n)

    out = (Path(args.output) if args.output
           else config.VIZ_DIR / f"routes_{in_path.stem}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"[viz_routes] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
