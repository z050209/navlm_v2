"""SV pano coverage + matched-POI map.

Layers on a single Leaflet map:

  1. **POI_BBOX** — the OSM extraction rectangle (3.8 x 3.9 km).
  2. **Matched POIs** — the VLM-named OSM POIs from `poi_scan.jsonl`
     (the anchors the targeted crawl was built around). Outliers
     whose geometry sits > OUTLIER_THRESHOLD_M from any SV pano are
     excluded from the layer and listed in `outlier_pois.json`.
  3. **SV panos** — every unique panorama in `meta.jsonl` (the
     GPS-ground-truth photos DINOv2 matches against). Blue dots.

  python -m src.viz_coverage   ->   viz/sv_coverage_map.html
"""

OUTLIER_THRESHOLD_M = 300.0     # POI geometry > this from any pano
                                # -> flagged as outlier and excluded

import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                           # noqa: E402
from src.poi_scan import poi_tier       # noqa: E402

TIER_COLOR = {1: "#d62728", 2: "#1f77b4", 3: "#7f7f7f"}


def build():
    import folium
    from shapely import wkt
    from shapely.ops import transform, unary_union
    from src.spatial import _proj_lonlat, _LAT_M

    # ── inputs ──────────────────────────────────────────────────────
    meta_path = config.STREETVIEW_DIR / "meta.jsonl"
    panos_seen = {}
    if meta_path.exists():
        for line in meta_path.open(encoding="utf-8"):
            m = json.loads(line)
            if m["pano_id"] not in panos_seen:
                panos_seen[m["pano_id"]] = (m["lat"], m["lon"])
    print(f"SV panos in meta.jsonl: {len(panos_seen)} unique")

    rec_path = config.CITY_DIR / "gps_recovery.jsonl"
    dino_pois = collections.Counter()
    if rec_path.exists():
        for line in rec_path.open(encoding="utf-8"):
            n = json.loads(line).get("dino_nearest_name", "")
            if n:
                dino_pois[n] += 1
    print(f"DINOv2-detected POIs (distinct): {len(dino_pois)}")

    pois = json.loads(
        (config.CITY_DIR / "pois.json").read_text(encoding="utf-8"))
    pois_map = {p["name"]: p for p in pois}

    scan_path = config.CITY_DIR / "poi_scan.jsonl"
    matched_counts = collections.Counter()
    if scan_path.exists():
        for line in scan_path.open(encoding="utf-8"):
            for mp in json.loads(line).get("matched", []):
                if mp.get("osm_name"):
                    matched_counts[mp["osm_name"]] += 1
    print(f"poi_scan matched POIs total: {len(matched_counts)}")

    # ── outlier flagging: POI whose entire geometry is >
    # OUTLIER_THRESHOLD_M from any SV pano. Catches VLM mis-guesses
    # for residential streets / mountain-area POIs that the videos
    # don't actually walk. Long features (Limmat, Zürichsee) are NOT
    # flagged because their geometry runs through the covered area.
    from shapely import wkt
    from shapely.geometry import Point, MultiPoint
    from shapely.ops import transform
    from src.spatial import _proj_lonlat, _LAT_M

    lat0_rad = math.radians((config.POI_BBOX[1] + config.POI_BBOX[3]) / 2)
    lon_m = _LAT_M * math.cos(lat0_rad)
    pano_pts_m = MultiPoint(
        [Point(lon * lon_m, lat * _LAT_M)
         for lat, lon in panos_seen.values()])

    inliers, outliers = {}, {}
    for name in matched_counts:
        p = pois_map.get(name)
        if not p or not p.get("geometry"):
            continue
        try:
            g_m = transform(_proj_lonlat, wkt.loads(p["geometry"]))
            d = float(g_m.distance(pano_pts_m))
        except Exception:
            continue
        (outliers if d > OUTLIER_THRESHOLD_M else inliers)[name] = d

    matched_names = set(inliers)
    print(f"  inliers (within {OUTLIER_THRESHOLD_M:.0f} m of a pano): "
          f"{len(inliers)}")
    print(f"  outliers (> {OUTLIER_THRESHOLD_M:.0f} m): {len(outliers)}")
    out_path = config.CITY_DIR / "outlier_pois.json"
    out_path.write_text(json.dumps(
        {name: round(d, 1) for name, d in
         sorted(outliers.items(), key=lambda x: -x[1])},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved outlier list to {out_path}")
    if outliers:
        print(f"  top outliers:")
        for name, d in sorted(outliers.items(),
                              key=lambda x: -x[1])[:5]:
            n_sightings = matched_counts[name]
            print(f"    {name:35s}  {d:6.0f} m  ({n_sightings} sightings)")

    # ── map ────────────────────────────────────────────────────────
    W, S, E, N = config.POI_BBOX
    fmap = folium.Map(location=[(S + N) / 2, (W + E) / 2],
                      zoom_start=15, tiles="OpenStreetMap")

    folium.Rectangle(bounds=[(S, W), (N, E)], color="#000", weight=2,
                     fill=False, tooltip="POI_BBOX").add_to(fmap)

    # (crawl-footprint polygon intentionally omitted — keep the map
    # simple. The POI_BBOX rectangle above is enough spatial context;
    # if needed we can derive the visited region from the matched POIs.)

    # Matched POIs (from poi_scan) — the VLM-named places the videos
    # visit. They're the *anchors* the targeted-crawl filter used.
    # Green to distinguish from DINOv2-detected POIs below.
    matched_layer = folium.FeatureGroup(
        name=(f"Matched POIs ({len(matched_names)} — VLM-named, "
              f"{len(outliers)} outliers excluded)"),
        show=True)
    for name, count in matched_counts.most_common():
        if name in outliers:                   # skip flagged outliers
            continue
        p = pois_map.get(name)
        if not p:
            continue
        popup = folium.Popup(
            f"<b>{name}</b><br>{p.get('kind_label', '?')}<br>"
            f"VLM matched this POI on <b>{count}</b> frames",
            max_width=320)
        tooltip = f"{name} ({count} frames)"
        try:
            g = wkt.loads(p.get("geometry", ""))
        except Exception:
            g = None
        if g is None or g.geom_type == "Point":
            folium.CircleMarker(
                location=[p["lat"], p["lon"]],
                radius=3 + math.log1p(count) * 2,
                color="#228833", fill=True, fill_opacity=0.75, weight=1,
                tooltip=tooltip, popup=popup,
            ).add_to(matched_layer)
        elif g.geom_type in ("LineString", "MultiLineString"):
            lines = ([g] if g.geom_type == "LineString"
                     else list(g.geoms))
            for ln in lines:
                folium.PolyLine(
                    locations=[(y, x) for x, y in ln.coords],
                    color="#228833", weight=2 + math.log1p(count) * 0.6,
                    opacity=0.75, tooltip=tooltip, popup=popup,
                ).add_to(matched_layer)
        # Polygons (water features) intentionally skipped, as before.
    matched_layer.add_to(fmap)

    # SV panos — BOUGHT (in meta.jsonl)
    sv_layer = folium.FeatureGroup(
        name=f"SV panos bought ({len(panos_seen)})",
        show=True)
    for pano_id, (lat, lon) in panos_seen.items():
        folium.CircleMarker(
            location=[lat, lon], radius=2, color="#4477AA",
            fill=True, fill_opacity=0.85, weight=0,
            tooltip=f"BOUGHT: {pano_id[:14]}").add_to(sv_layer)
    sv_layer.add_to(fmap)

    # SV panos — AVAILABLE but NOT bought (panos.jsonl minus meta.jsonl).
    # Lets the user compare the targeted 1,108 against everything Google
    # has in the bbox (1,915), and see what we deliberately skipped.
    panos_avail_path = config.STREETVIEW_DIR / "panos.jsonl"
    all_panos = {}
    if panos_avail_path.exists():
        for line in panos_avail_path.open(encoding="utf-8"):
            m = json.loads(line)
            all_panos[m["pano_id"]] = (m["lat"], m["lon"])
    not_bought = {pid: ll for pid, ll in all_panos.items()
                  if pid not in panos_seen}
    print(f"available panos (from panos.jsonl): {len(all_panos)}")
    print(f"  not bought (available - bought): {len(not_bought)}")
    nb_layer = folium.FeatureGroup(
        name=(f"Available but NOT bought ({len(not_bought)} "
              f"— Google has, we skipped)"),
        show=False)         # off by default to keep the map clean
    for pid, (lat, lon) in not_bought.items():
        folium.CircleMarker(
            location=[lat, lon], radius=2, color="#cc3333",
            fill=True, fill_opacity=0.75, weight=0,
            tooltip=f"AVAILABLE (skipped): {pid[:14]}",
        ).add_to(nb_layer)
    nb_layer.add_to(fmap)

    # (DINOv2-detected POI layer intentionally omitted for now — it
    # would be the *stale* count from the 178-pano gps_recovery run.
    # Will be added back after we re-run gps_recovery on the full
    # 1,108-pano SV set.)

    folium.LayerControl(collapsed=False).add_to(fmap)

    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; background: white;
         padding: 10px 12px; border: 1px solid #888; border-radius: 6px;
         font-family: Arial, sans-serif; font-size: 12px; z-index: 9999;
         box-shadow: 0 1px 4px rgba(0,0,0,0.2); max-width: 440px;">
      <b>SV coverage + matched POIs</b><br>
      Black rectangle: <b>POI_BBOX</b> (OSM extraction region,
      &asymp; 3.8 &times; 3.9 km).<br>
      <span style="color:#228833;">&#9679;</span>
      <b>Matched POIs</b> ({len(matched_names)}) &mdash; VLM-named
      via poi_scan; the anchors the targeted crawl was built around.
      Points as dots, streets as polylines along the actual road.
      Size / weight &prop; log(frames+1).
      ({len(outliers)} POIs whose geometry sits &gt;
      {OUTLIER_THRESHOLD_M:.0f} m from any pano are excluded &mdash;
      see <code>data/cities/zurich/outlier_pois.json</code>.)<br>
      <span style="color:#4477AA;">&#9679;</span>
      <b>SV panos bought</b> ({len(panos_seen)}) &mdash; the
      GPS-known reference photos DINOv2 matches against.<br>
      <span style="color:#cc3333;">&#9679;</span>
      <b>Available but not bought</b> ({len(not_bought)}, toggle on)
      &mdash; red dots: panos Google has in the bbox that we
      deliberately skipped (outside the 150 m POI buffer). Toggle on
      to see where we left gaps.<br>
      <i>DINOv2-detected POI layer omitted for now &mdash; will be
      added back after the next gps_recovery run on the full
      1,108-pano set.</i>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))

    out = config.VIZ_DIR / "sv_coverage_map.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
