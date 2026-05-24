"""POI-scan visualization — map of the matched POIs + the derived
Street View crawl bboxes, with the few extended-feature outliers that
pull the *raw* bbox out of central Zurich marked distinctly.

    python -m src.viz_scan      ->   viz/poi_scan_map.html
"""

import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                # noqa: E402
from src.poi_scan import poi_tier            # noqa: E402
from src.streetview import bbox_from_pois    # noqa: E402

# tier -> colour (L1 iconic = red, L2 supporting = blue, L3 = grey)
TIER_COLOR = {1: "#d62728", 2: "#1f77b4", 3: "#7f7f7f"}


def _gather():
    """Returns (matched_pois, outliers, clean_bbox, raw_bbox).
    A POI is an *outlier* if its OSM centroid lies outside config.POI_BBOX
    — typically a river / lake / very long street whose centroid is far
    from central Zurich and would balloon the crawl bbox."""
    pois = json.loads(
        (config.CITY_DIR / "pois.json").read_text(encoding="utf-8"))
    pmap = {p["name"]: p for p in pois}
    scan = [json.loads(l) for l in
            (config.CITY_DIR / "poi_scan.jsonl").open(encoding="utf-8")
            if l.strip()]

    n_sight = collections.Counter()
    example = {}              # osm_name -> (video, frame_id, reasoning)
    for r in scan:
        for m in r["matched"]:
            n = m["osm_name"]
            n_sight[n] += 1
            if n not in example:
                example[n] = (r["video"], r["frame_id"], r["reasoning"])

    W, S, E, N = config.POI_BBOX
    matched, outliers = [], []
    for name, count in n_sight.items():
        p = pmap.get(name)
        if not p:
            continue
        rec = {
            "name": name, "lat": p["lat"], "lon": p["lon"],
            "kind_label": p.get("kind_label", "?"),
            "osm_kind": p.get("osm_kind", ""),
            "tier": poi_tier(p.get("osm_kind", "")),
            "n": count,
            "ex_video": example[name][0], "ex_frame": example[name][1],
            "reasoning": example[name][2],
        }
        if W <= p["lon"] <= E and S <= p["lat"] <= N:
            matched.append(rec)
        else:
            outliers.append(rec)

    pts_clean = [(r["lat"], r["lon"]) for r in matched]
    pts_raw = pts_clean + [(r["lat"], r["lon"]) for r in outliers]
    return (matched, outliers,
            bbox_from_pois(pts_clean, margin_m=300),
            bbox_from_pois(pts_raw, margin_m=300))


def build():
    import folium
    matched, outliers, clean_bbox, raw_bbox = _gather()
    W, S, E, N = config.POI_BBOX

    m = folium.Map(location=[(S + N) / 2, (W + E) / 2], zoom_start=14,
                   tiles="OpenStreetMap")

    # bbox rectangles — folium bounds are [[s,w],[n,e]]
    def rect(b, color, weight, dash, name):
        w, s, e, n = b
        folium.Rectangle(
            bounds=[(s, w), (n, e)], color=color, weight=weight,
            dash_array=dash, fill=False,
            tooltip=name).add_to(m)

    rect(config.POI_BBOX, "#000", 2, None,
         "POI_BBOX — OSM extraction region (central Zurich)")
    rect(clean_bbox, "#d62728", 3, None,
         "Crawl bbox + 300 m (centroid-clipped — recommended)")
    rect(raw_bbox, "#888888", 2, "8,8",
         "Raw scan bbox + 300 m (pulled by extended-feature outliers)")

    # matched POIs as size-by-sightings circles
    for r in matched:
        radius = 3 + math.log1p(r["n"]) * 2.5
        popup = folium.Popup(
            f"<b>{r['name']}</b><br>{r['kind_label']} (L{r['tier']})<br>"
            f"<b>{r['n']}</b> sightings<br>"
            f"<i>{r['ex_video']}/{r['ex_frame']}:</i><br>"
            f"<small>{r['reasoning'][:240]}</small>",
            max_width=320)
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=radius,
            color=TIER_COLOR[r["tier"]], fill=True, fill_opacity=0.7,
            weight=1, popup=popup,
            tooltip=f"{r['name']} ({r['n']})").add_to(m)

    # outliers — distinct orange flag
    for r in outliers:
        popup = folium.Popup(
            f"<b>{r['name']}</b> — OUTLIER<br>"
            f"{r['kind_label']} (L{r['tier']})<br>"
            f"OSM centroid is outside central Zurich, so this POI is "
            f"excluded from the clean crawl bbox.<br>"
            f"<b>{r['n']}</b> sightings", max_width=320)
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=folium.Icon(color="orange", icon="warning-sign"),
            popup=popup,
            tooltip=f"OUTLIER: {r['name']} ({r['n']})").add_to(m)

    # legend (fixed HTML overlay)
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; background: white;
         padding: 10px 12px; border: 1px solid #888; border-radius: 6px;
         font-family: Arial, sans-serif; font-size: 12px; z-index: 9999;
         box-shadow: 0 1px 4px rgba(0,0,0,0.2);">
      <b>POI scan — {len(matched)} matched POIs</b> &nbsp;
      ({len(outliers)} outliers)<br>
      <span style="color:#d62728; font-size:16px;">&#9679;</span> L1 iconic landmark&nbsp;
      <span style="color:#1f77b4; font-size:16px;">&#9679;</span> L2 supporting&nbsp;
      <span style="color:#7f7f7f; font-size:16px;">&#9679;</span> L3 other<br>
      <span style="color:orange; font-size:14px;">&#9650;</span>
      Outlier (extended-feature centroid)&nbsp;·&nbsp;
      dot size &prop; log(sightings + 1)<br>
      <span style="color:#d62728;"><b>━</b></span> crawl bbox + 300 m (clean)&nbsp;
      <span style="color:#888;"><b>┄┄</b></span> raw bbox + 300 m&nbsp;
      <span style="color:#000;"><b>━</b></span> POI_BBOX (OSM extraction)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    out = config.VIZ_DIR / "poi_scan_map.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"wrote {out}")
    print(f"  matched POIs:  {len(matched)} (clean) + "
          f"{len(outliers)} outliers")
    print(f"  clean bbox + 300 m (W,S,E,N): "
          f"({clean_bbox[0]:.5f}, {clean_bbox[1]:.5f}, "
          f"{clean_bbox[2]:.5f}, {clean_bbox[3]:.5f})")
    print(f"  raw   bbox + 300 m (W,S,E,N): "
          f"({raw_bbox[0]:.5f}, {raw_bbox[1]:.5f}, "
          f"{raw_bbox[2]:.5f}, {raw_bbox[3]:.5f})")


if __name__ == "__main__":
    build()
