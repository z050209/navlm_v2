"""Plot the GPS paths of all walking-tour videos on one interactive map.

For each video, walks frames in order, takes GPS where available, drops
low-confidence/missing, smooths a bit, draws a coloured polyline. Iconic
POIs from the OSM table are dropped as small markers for context.

Output: data/cities/zurich/_paths_map.html (open in browser, or click
"Map" link in the synth_viewer).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import folium
from folium import plugins
import networkx as nx
import pickle as _pickle

ROOT = Path(__file__).resolve().parent.parent
COLORS = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6",
          "#ec4899", "#06b6d4", "#f97316"]


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def frame_idx(fid):
    m = re.search(r"(\d+)", fid)
    return int(m.group(1)) if m else -1


def gps_track_for_video(gps_jsonl_path):
    """Read frame_gps* file, return ordered list of (frame_id, lat, lon).

    Priority:
      1. HMM-matched output (snap_lat, snap_lon) — vouched for OSM topology
      2. Refined visual-match high/medium (gps_v2)
      3. Raw visual-match high/medium (gps)
    """
    rows = load_jsonl(gps_jsonl_path)
    rows.sort(key=lambda r: frame_idx(r["frame_id"]))
    track = []
    for r in rows:
        if r.get("snap_lat") is not None and r.get("snap_lon") is not None and r.get("matched"):
            track.append((r["frame_id"], r["snap_lat"], r["snap_lon"]))
        elif r.get("confidence_v2") in ("high", "medium") and r.get("gps_v2"):
            track.append((r["frame_id"], r["gps_v2"][0], r["gps_v2"][1]))
        elif r.get("confidence") in ("high", "medium") and r.get("gps"):
            track.append((r["frame_id"], r["gps"][0], r["gps"][1]))
    return track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output html path (default depends on --variant)")
    ap.add_argument("--density", type=int, default=150,
                    help="target points per video polyline (higher = more detail, "
                         "but heavier rendering)")
    ap.add_argument("--graph", default="data/cities/zurich/osm_walking.pkl",
                    help="OSM walking graph for routing line segments")
    ap.add_argument("--no-osm-routing", action="store_true",
                    help="draw straight-line segments instead of OSM-routed paths")
    ap.add_argument("--variant", default="",
                    help="namespace suffix to read pipeline_<variant>/<video>/step_08_hmm.jsonl "
                         "(e.g. '_hq' for HQ Mapillary). Empty = legacy paths.")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"data/cities/zurich/_paths_map{args.variant}.html"

    # Load OSM walking graph for routing
    osm_G = None
    if not args.no_osm_routing and Path(args.graph).exists():
        print(f"[map] loading OSM graph for routing")
        with open(args.graph, "rb") as fh:
            osm_G = _pickle.load(fh)
        if osm_G.is_directed():
            osm_G = osm_G.to_undirected()
        # Build a fast nearest-node lookup
        try:
            import osmnx as _ox
            print(f"[map] {osm_G.number_of_nodes()} nodes — using osmnx for snap")
        except Exception:
            osm_G = None

    sources = []  # (label, path-jsonl, color)
    extra_root = ROOT / "data/cities/zurich"

    if args.variant:
        # Variant mode: read from pipeline_<variant>/<canonical>/step_08_hmm.jsonl
        # Maps each canonical video name to a path under the new HQ pipeline.
        sys.path.insert(0, str(ROOT / "pipeline"))
        from config import VIDEOS  # noqa
        pipeline_dir = extra_root / f"pipeline{args.variant}"
        for i, (canonical, _, _) in enumerate(VIDEOS):
            hmm = pipeline_dir / canonical / "step_08_hmm.jsonl"
            if hmm.exists():
                label = f"{canonical} (HMM-{args.variant.lstrip('_')})"
                sources.append((label, hmm, COLORS[i % len(COLORS)]))
    else:
        # Legacy mode: read existing extra_*_frame_gps_matched.jsonl files
        # Prefer HMM-matched outputs when available; else refined.
        matched = extra_root / "frame_gps_matched.jsonl"
        final   = extra_root / "frame_gps_final.jsonl"
        if matched.exists():
            sources.append(("Original (HMM)", matched, COLORS[0]))
        elif final.exists():
            sources.append(("Original", final, COLORS[0]))

        seen_basenames = set()
        for f in sorted(extra_root.glob("extra_*_frame_gps_matched.jsonl")):
            m = re.match(r"extra_(.+)_frame_gps_matched\.jsonl", f.name)
            if m:
                base = m.group(1)
                label = base[:32] + " (HMM)"
                sources.append((label, f, COLORS[len(sources) % len(COLORS)]))
                seen_basenames.add(base)
        for f in sorted(extra_root.glob("extra_*_frame_gps_refined.jsonl")):
            m = re.match(r"extra_(.+)_frame_gps_refined\.jsonl", f.name)
            if m and m.group(1) not in seen_basenames:
                label = m.group(1)[:32] + " (refined)"
                sources.append((label, f, COLORS[len(sources) % len(COLORS)]))
                seen_basenames.add(m.group(1))

    print(f"[map] {len(sources)} videos")

    # Build map
    m = folium.Map(location=[47.374, 8.541], zoom_start=15,
                   tiles="OpenStreetMap")

    overall_pts = []
    for vid_idx, (label, path, color) in enumerate(sources):
        track = gps_track_for_video(path)
        if not track:
            print(f"  [skip] {label} — no high/medium-conf points")
            continue
        # Down-sample to ~density points
        if len(track) > args.density:
            step = max(1, len(track) // args.density)
            track = track[::step]
        coords = [(lat, lon) for _, lat, lon in track]
        overall_pts.extend(coords)

        # OSM-routed dense polyline: between consecutive sampled points,
        # use shortest path on the walking graph so the line follows real
        # streets (not straight-line jumps).
        routed_coords = coords
        if osm_G is not None and len(coords) > 1:
            import osmnx as _ox
            routed = []
            prev_node = None
            for i, (lat, lon) in enumerate(coords):
                try:
                    cur_node = _ox.distance.nearest_nodes(osm_G, lon, lat)
                except Exception:
                    cur_node = None
                if i == 0 or prev_node is None or cur_node is None or cur_node == prev_node:
                    routed.append((lat, lon))
                else:
                    try:
                        path_nodes = nx.shortest_path(
                            osm_G, prev_node, cur_node, weight="length")
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        path_nodes = []
                    if path_nodes:
                        for n in path_nodes:
                            routed.append((osm_G.nodes[n]["y"], osm_G.nodes[n]["x"]))
                    else:
                        routed.append((lat, lon))  # fallback straight-line
                prev_node = cur_node
            if len(routed) > 1:
                routed_coords = routed

        # Per-video FeatureGroup so user can toggle layers
        fg = folium.FeatureGroup(name=f"{label} ({len(coords)} pts)",
                                  show=True)

        # Animated direction-aware path along the OSM-routed geometry
        plugins.AntPath(
            locations=routed_coords, color=color, weight=5, opacity=0.85,
            delay=1500, dash_array=[12, 28],
            popup=f"{label} ({len(coords)} sample pts, routed = {len(routed_coords)})",
        ).add_to(fg)

        # Start = green, End = red, with sequence number in tooltip
        folium.CircleMarker(
            coords[0], radius=8, color="#10b981", fill=True,
            fill_opacity=0.95, weight=3,
            popup=f"<b>{label}</b><br>START (frame {track[0][0]})",
            tooltip=f"▶ {label} START",
        ).add_to(fg)
        folium.CircleMarker(
            coords[-1], radius=8, color="#ef4444", fill=True,
            fill_opacity=0.95, weight=3,
            popup=f"<b>{label}</b><br>END (frame {track[-1][0]})",
            tooltip=f"⏹ {label} END",
        ).add_to(fg)

        # Drop ~6 evenly-spaced numbered checkpoints along the path
        n_checks = min(6, len(coords) - 2)
        if n_checks > 0:
            stride = max(1, len(coords) // (n_checks + 1))
            for i in range(1, n_checks + 1):
                idx = i * stride
                if idx >= len(coords) - 1:
                    break
                fid = track[idx][0]
                folium.CircleMarker(
                    coords[idx], radius=4, color=color, fill=True,
                    fill_color="#fff", fill_opacity=0.95, weight=2,
                    tooltip=f"{label} #{i}/{n_checks} ({fid})",
                ).add_to(fg)

        fg.add_to(m)
        print(f"  [{label}] {len(coords)} polyline points  color={color}")

    # ─────── POIs ───────
    pois_path = ROOT / "data/cities/zurich/landmarks_zurich_osm.json"
    pois = {}
    if pois_path.exists():
        with open(pois_path) as f:
            pois = json.load(f)
        import sys
        sys.path.insert(0, str(ROOT / "toolbox"))
        from synth.sampling import poi_tier

        # OSM tier-1 POIs (purple stars) — iconic landmarks only.
        # tier-2/3 layers were removed — too cluttered, low signal.
        fg_t1 = folium.FeatureGroup(name="★ OSM tier-1 (iconic)", show=True)
        n_t1 = 0
        for name, p in pois.items():
            if poi_tier(name) != 1:
                continue
            tip = f"{name} ({p.get('kind_label','')})"
            folium.Marker([p["lat"], p["lon"]],
                icon=folium.Icon(color="purple", icon="star"),
                popup=tip, tooltip=name).add_to(fg_t1)
            n_t1 += 1
        fg_t1.add_to(m)
        print(f"[map] {n_t1} tier-1 POIs marked")

        # Layer 2: per-video POIs touched (using HMM-matched GPS)
        # For each video's full track, find POIs within 50m of any matched
        # frame. Mark with the video's colour so we can see who went where.
        from scipy.spatial import cKDTree
        import numpy as np
        LAT_M, LON_M = 111000, 78000
        poi_names_all = list(pois.keys())
        poi_xy = np.array([[pois[n]["lat"]*LAT_M, pois[n]["lon"]*LON_M]
                            for n in poi_names_all])
        poi_tree = cKDTree(poi_xy)

        # Map source jsonl → frames-dir basename (so the viewer can serve
        # a sample image from the right video)
        def frames_subdir_for(source_path_name):
            # e.g. "extra_<name>_frame_gps_matched.jsonl" → "extra_<name>"
            n = source_path_name.replace("_frame_gps_matched.jsonl", "")
            n = n.replace("_frame_gps_refined.jsonl", "")
            n = n.replace("_frame_gps_final.jsonl", "")
            n = n.replace("_frame_gps.jsonl", "")
            if n in ("frame_gps_matched", "frame_gps_final", "frame_gps"):
                return "zurich"  # original video
            return n

        for vid_idx, (label, path, color) in enumerate(sources):
            track_full = gps_track_for_video(path)
            if not track_full:
                continue
            # For each POI, find the CLOSEST frame from this video
            poi_best_frame = {}    # poi → (frame_id, distance)
            poi_count = {}
            for fid, lat, lon in track_full:
                d_m, ix = poi_tree.query([lat * LAT_M, lon * LON_M], k=1)
                if d_m <= 50:
                    name = poi_names_all[int(ix)]
                    poi_count[name] = poi_count.get(name, 0) + 1
                    if name not in poi_best_frame or d_m < poi_best_frame[name][1]:
                        poi_best_frame[name] = (fid, d_m)
            if not poi_best_frame:
                continue

            fg_v = folium.FeatureGroup(
                name=f"   POIs touched by {label}", show=False)
            sub = frames_subdir_for(path.name)
            for name, (fid, d_m) in sorted(poi_best_frame.items(),
                                             key=lambda x: -poi_count[x[0]])[:30]:
                p = pois[name]
                tier = poi_tier(name)
                kind = p.get("kind_label", "")
                cnt = poi_count[name]
                # On-hover tooltip: name + thumbnail + description
                tooltip_html = (
                    f"<div style='max-width:300px;font-size:13px;'>"
                    f"<div style='font-weight:600;font-size:14px;'>{name}</div>"
                    f"<div style='color:#666;'>{kind}</div>"
                    f"<img src='/frame_image/{sub}/{fid}.jpg' "
                    f"style='width:280px;margin-top:6px;border-radius:4px;' "
                    f"onerror=\"this.style.display='none'\"/>"
                    f"<div style='margin-top:4px;color:#888;font-size:11px;'>"
                    f"{label} touched {cnt}× · closest frame {fid} ({d_m:.0f}m away)"
                    f"</div></div>"
                )
                # Star icon — DivIcon with colored star
                star_html = (
                    f"<div style='font-size:26px;line-height:24px;"
                    f"color:{color};text-shadow:0 0 3px black,0 0 1px black;"
                    f"text-align:center;'>★</div>"
                )
                folium.Marker(
                    [p["lat"], p["lon"]],
                    icon=folium.DivIcon(html=star_html, icon_size=(26, 26),
                                          icon_anchor=(13, 13)),
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=tooltip_html,
                ).add_to(fg_v)
            fg_v.add_to(m)
            print(f"  [{label}] {len(poi_best_frame)} POIs touched (HMM-snapped)")

        # ─────── VLM-confirmed POIs per video (independent visual evidence)
        # Look up POI coordinates from OSM POI table OR scenery_pois.py
        try:
            sys.path.insert(0, str(ROOT / "toolbox"))
            from scenery_pois import SCENERY_POIS
        except ImportError:
            SCENERY_POIS = {}

        def poi_coords(name):
            if name in pois:
                return (pois[name]["lat"], pois[name]["lon"], pois[name].get("kind_label", ""))
            if name in SCENERY_POIS:
                s = SCENERY_POIS[name]
                return (s["lat"], s["lon"], s.get("kind_label", s.get("kind", "")))
            return None

        # Load VLM multi-scan output
        vlm_path = ROOT / "data/cities/zurich/_video_poi_multi.jsonl"
        if vlm_path.exists():
            vlm_per_video_frames = {}   # video → poi → list of frame_ids
            for ln in open(vlm_path):
                r = json.loads(ln)
                if not r.get("visible_pois"): continue
                for p in r["visible_pois"]:
                    vlm_per_video_frames.setdefault(r["video"], {}) \
                        .setdefault(p, []).append(r["frame_id"])

            # The multi-scan stores labels via d.name.replace("extra_", "")[:55]
            # but here `label` is base[:32] + " (HMM)". Match by prefix.
            def find_vlm_key(label):
                stem = label.replace(" (HMM)", "").replace(" (refined)", "").strip()
                if stem in vlm_per_video_frames:
                    return stem
                # try prefix match (32-char truncation case)
                for k in vlm_per_video_frames:
                    if k.startswith(stem):
                        return k
                # special: "Original" maps to "Original" or "Original (HMM)" already
                if stem == "Original" and "Original" in vlm_per_video_frames:
                    return "Original"
                return None

            for vid_idx, (label, path, color) in enumerate(sources):
                vk = find_vlm_key(label)
                vlm_pois = vlm_per_video_frames.get(vk, {}) if vk else {}
                if not vlm_pois:
                    continue
                fg_vlm = folium.FeatureGroup(
                    name=f"   VLM-confirmed POIs in {label}", show=False)
                sub = frames_subdir_for(path.name)
                for poi_name, frames_list in sorted(vlm_pois.items(),
                                                       key=lambda x: -len(x[1])):
                    coords = poi_coords(poi_name)
                    if not coords:
                        continue
                    plat, plon, kind = coords
                    cnt = len(frames_list)
                    sample_fid = frames_list[len(frames_list) // 2]
                    tooltip_html = (
                        f"<div style='max-width:300px;font-size:13px;'>"
                        f"<div style='font-weight:600;font-size:14px;'>"
                        f"{poi_name} <span style='color:#10b981'>✓ VLM</span></div>"
                        f"<div style='color:#666;'>{kind}</div>"
                        f"<img src='/frame_image/{sub}/{sample_fid}.jpg' "
                        f"style='width:280px;margin-top:6px;border-radius:4px;' "
                        f"onerror=\"this.style.display='none'\"/>"
                        f"<div style='margin-top:4px;color:#888;font-size:11px;'>"
                        f"VLM saw {poi_name} in {cnt} of {label}'s sampled frames"
                        f"</div></div>"
                    )
                    # Diamond marker (◆) — distinct from GPS-touched ★ stars
                    diamond_html = (
                        f"<div style='font-size:24px;line-height:22px;"
                        f"color:{color};text-shadow:0 0 4px white,0 0 1px black;"
                        f"text-align:center;font-weight:700;'>◆</div>"
                    )
                    folium.Marker(
                        [plat, plon],
                        icon=folium.DivIcon(html=diamond_html, icon_size=(24, 24),
                                              icon_anchor=(12, 12)),
                        tooltip=folium.Tooltip(tooltip_html, sticky=True),
                        popup=tooltip_html,
                    ).add_to(fg_vlm)
                fg_vlm.add_to(m)
                print(f"  [{label}] {len(vlm_pois)} VLM-confirmed POIs")

    # ─────── Mapillary coverage bboxes ───────
    # zurich/ 5k (old town), zurich_full 89k (wide), and the planned
    # zurich_dense10k tile-fetch box.
    bbox_zurich_5k        = ((47.366, 8.528), (47.385, 8.557))
    bbox_zurich_full      = ((47.340, 8.480), (47.420, 8.600))
    bbox_zurich_dense10k  = ((47.364, 8.528), (47.386, 8.560))   # planned new (2.5km×2.5km, shifted east)

    fg_bbox = folium.FeatureGroup(name="Mapillary coverage bboxes", show=True)
    folium.Rectangle(
        bounds=bbox_zurich_5k, color="#56d364", weight=3,
        fill=False, opacity=0.85,
        popup="Mapillary 5k (old-town focus, already downloaded)",
        tooltip="Mapillary 5k bbox · 5000 imgs · already downloaded"
    ).add_to(fg_bbox)
    folium.Rectangle(
        bounds=bbox_zurich_full, color="#3b82f6", weight=2,
        fill=False, opacity=0.6, dash_array="6,8",
        popup="Mapillary full 89k (wider Zurich)",
        tooltip="Mapillary 89k bbox · 89k imgs"
    ).add_to(fg_bbox)
    folium.Rectangle(
        bounds=bbox_zurich_dense10k, color="#000000", weight=4,
        fill=True, fill_color="#000000", fill_opacity=0.05,
        opacity=0.95,
        popup="<b>NEW dense-tile fetch (~10-30k imgs)</b><br>"
              "BBOX = (8.528, 47.364, 8.560, 47.386)<br>"
              "2.5 km × 2.5 km · tiles = 200m × 200m",
        tooltip="★ NEW planned fetch · 2.5km × 2.5km · 200m tiles"
    ).add_to(fg_bbox)
    fg_bbox.add_to(m)
    print(f"[map] Mapillary coverage bboxes added (incl. new dense-fetch box)")

    # Fit map to data
    if overall_pts:
        sw = (min(p[0] for p in overall_pts), min(p[1] for p in overall_pts))
        ne = (max(p[0] for p in overall_pts), max(p[1] for p in overall_pts))
        m.fit_bounds([sw, ne])

    # Layer toggle so each video can be hidden/shown
    folium.LayerControl(collapsed=False).add_to(m)

    # Big readable layer control + select all/none buttons
    big_css = """
    <style>
      .leaflet-control-layers, .leaflet-control-layers-expanded {
        font-size: 18px !important;
        line-height: 1.6 !important;
        padding: 12px 14px !important;
        max-height: 75vh !important;
        overflow-y: auto !important;
        min-width: 320px !important;
      }
      .leaflet-control-layers label { padding: 4px 0 !important; }
      .leaflet-control-layers-base label,
      .leaflet-control-layers-overlays label { font-size: 18px !important; }
      .leaflet-control-layers input[type=checkbox],
      .leaflet-control-layers input[type=radio] {
        transform: scale(1.4); margin-right: 8px;
      }
      #layer-bulk {
        padding: 6px 10px; border-bottom: 1px solid #ddd;
        background: #fafafa; display: flex; gap: 6px;
        position: sticky; top: 0; z-index: 1;
      }
      #layer-bulk button {
        font-size: 13px; padding: 4px 10px; border-radius: 4px;
        border: 1px solid #999; background: white; cursor: pointer;
      }
      #layer-bulk button:hover { background: #eaeaea; }
    </style>
    <script>
    function _layerBulk(checked) {
      const root = document.querySelector('.leaflet-control-layers-overlays');
      if (!root) return;
      root.querySelectorAll('input[type=checkbox]').forEach(cb => {
        if (cb.checked !== checked) cb.click();
      });
    }
    function _addLayerBulkButtons() {
      const overlays = document.querySelector('.leaflet-control-layers-overlays');
      if (!overlays || document.getElementById('layer-bulk')) return;
      const bar = document.createElement('div');
      bar.id = 'layer-bulk';
      bar.innerHTML =
        '<button onclick="_layerBulk(true)">✓ Select all</button>' +
        '<button onclick="_layerBulk(false)">✗ Unselect all</button>';
      overlays.parentNode.insertBefore(bar, overlays);
    }
    // Folium injects layer control after page load; poll briefly
    let _bulk_tries = 0;
    const _bulk_iv = setInterval(() => {
      _addLayerBulkButtons();
      if (document.getElementById('layer-bulk') || ++_bulk_tries > 30) {
        clearInterval(_bulk_iv);
      }
    }, 300);
    </script>
    """
    m.get_root().html.add_child(folium.Element(big_css))

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:10px 14px;border-radius:6px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:sans-serif;
                font-size:13px;">
      <b>Legend</b><br>
      <span style="color:#10b981">●</span> walk start &nbsp;
      <span style="color:#ef4444">●</span> walk end<br>
      <span style="color:purple">★</span> tier-1 POI<br>
      <span>○ checkpoint (small white-filled marker, in walk order)</span><br>
      <small>Arrows on each path animate in walking direction.</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"\n[map] → {out}")
    print(f"      open in browser, or http://<host>:9000/map  (link from viewer)")


if __name__ == "__main__":
    main()
