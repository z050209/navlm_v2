"""Folium map of the 89 unique panos in the matched cohort.

Each pano = one circle:
  - position: pano's GT GPS (from SV meta)
  - size:     proportional to # of matched frames at this pano
  - colour:   by # of distinct attractions tagged at this pano
              (1 = blue, 2-3 = green, 4+ = orange, 8+ = red)
  - popup:    pano_id, #frames, attractions list

Overlay: the 21 canonical attraction centroids as red star markers
        (so we can see how the panos cluster around them).

  python -m src.a2_viz_map
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402
from src.a2_attraction_slots import ATTRACTIONS_21  # noqa: E402


CANON = {en for en, *_ in ATTRACTIONS_21}


def main():
    import folium

    # ── load matched cohort + frame→GPS lookup ──────────────────────
    matched_rows = [json.loads(l) for l in
                    (config.CITY_DIR / "a2"
                     / "GPS_VLM_GEO.jsonl").open(encoding="utf-8")
                    if l.strip() and json.loads(l)["matched"]]

    gps_lookup = {}
    for line in (config.CITY_DIR
                 / "gps_recovery_full.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("g_dino") and r.get("top_sv_id"):
            pano = (r["top_sv_id"].rsplit("_h", 1)[0]
                    if "_h" in r["top_sv_id"] else r["top_sv_id"])
            gps_lookup[(r["video"], r["frame_id"])] = {
                "gps": tuple(r["g_dino"]),
                "pano_id": pano,
            }

    def frame_attractions(r):
        out = set()
        for a in r["list_a_gps"].get("attractions", []):
            if a in CANON: out.add(a)
        for a in r["list_b_vlm"].get("attractions", []):
            if a in CANON: out.add(a)
        for m in r["matches"]:
            for nm in [m["gps_name"], m["vlm_name"]]:
                if nm in CANON: out.add(nm)
        return out

    # aggregate per pano
    pano_info = collections.defaultdict(
        lambda: {"frames": 0, "attractions": set(),
                  "gps": None, "videos": collections.Counter()})
    for r in matched_rows:
        key = (r["video"], r["frame_id"])
        g = gps_lookup.get(key)
        if not g:
            continue
        info = pano_info[g["pano_id"]]
        info["gps"] = g["gps"]
        info["frames"] += 1
        info["videos"][r["video"]] += 1
        info["attractions"] |= frame_attractions(r)

    print(f"[map] unique panos in matched cohort: {len(pano_info)}")

    # ── build the map, centred on Zurich old town ───────────────────
    m = folium.Map(location=[47.37, 8.541], zoom_start=15,
                    tiles="OpenStreetMap")

    # 21 canonical attractions — red stars
    attr_fg = folium.FeatureGroup(name="21 canonical attractions",
                                   show=True)
    for en, zh, lat, lon, kind in ATTRACTIONS_21:
        folium.Marker(
            [lat, lon],
            tooltip=f"{en} ({zh})",
            popup=folium.Popup(
                f"<b>{en}</b><br>{zh}<br>kind: {kind}",
                max_width=240),
            icon=folium.Icon(color="red", icon="star", prefix="fa"),
        ).add_to(attr_fg)
    attr_fg.add_to(m)

    # 89 panos — circles
    pano_fg = folium.FeatureGroup(name="89 matched-cohort panos",
                                   show=True)
    for pid, info in pano_info.items():
        n_a = len(info["attractions"])
        n_f = info["frames"]
        if n_a >= 8:
            colour = "#c0392b"      # red — see-everything panos
        elif n_a >= 4:
            colour = "#e67e22"      # orange
        elif n_a >= 2:
            colour = "#27ae60"      # green
        else:
            colour = "#2980b9"      # blue — single-attraction pano

        attrs_html = ", ".join(sorted(info["attractions"])) or "(none)"
        videos_html = ", ".join(
            f"{v}({c})" for v, c in info["videos"].most_common())
        popup_html = (
            f"<b>pano</b> {pid}<br>"
            f"<b>frames:</b> {n_f}<br>"
            f"<b>attractions ({n_a}):</b><br>"
            f"&nbsp;&nbsp;{attrs_html}<br>"
            f"<b>videos:</b><br>"
            f"&nbsp;&nbsp;{videos_html}")
        folium.CircleMarker(
            location=info["gps"],
            radius=4 + min(n_f / 8.0, 12),    # cap at 16px
            color=colour, fill=True, fill_color=colour, fill_opacity=0.65,
            weight=2,
            tooltip=f"{n_f} frames · {n_a} attractions",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(pano_fg)
    pano_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # legend (a small HTML overlay)
    legend = """
    <div style="position: fixed; bottom: 20px; left: 20px;
                background: white; padding: 10px 14px; border-radius: 8px;
                border: 1px solid #bbb; font: 12px sans-serif; z-index: 9999;
                box-shadow: 0 2px 6px rgba(0,0,0,.15);">
      <div style="font-weight:600;margin-bottom:6px;">
        89 matched-cohort GPS spots
      </div>
      <div><span style="color:#c0392b">●</span> 8+ attractions visible</div>
      <div><span style="color:#e67e22">●</span> 4-7 attractions</div>
      <div><span style="color:#27ae60">●</span> 2-3 attractions</div>
      <div><span style="color:#2980b9">●</span> 1 attraction (singleton)</div>
      <div style="margin-top:6px;color:#888;">
        circle size ∝ # of matched video frames at the pano
      </div>
      <div style="margin-top:4px;">
        <span style="color:red;font-weight:600;">★</span>
        canonical attraction GPS (21 total)
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    out = Path("viz/a2_mapped_GPS_spot.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"[map] wrote {out.resolve()}")
    print(f"      file:///{str(out.resolve()).replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
