"""GPS recovery visualization — every frame's accepted GPS on a Leaflet
map, with toggle layers for the rejected categories so the SV coverage
gap is immediately visible.

  python -m src.viz_recovery   ->   viz/gps_recovery_map.html
"""

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# colour-blind-friendly per-video palette (Tol Bright); cycles if > 8.
VIDEO_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44",
                "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]


def build():
    import folium
    rows = [json.loads(l) for l in
            (config.CITY_DIR / "gps_recovery.jsonl").open(encoding="utf-8")
            if l.strip()]
    if not rows:
        sys.exit("no gps_recovery.jsonl — run `python -m src.gps_recovery`")

    videos = sorted({r["video"] for r in rows})
    vcolor = {v: VIDEO_COLORS[i % len(VIDEO_COLORS)]
              for i, v in enumerate(videos)}

    W, S, E, N = config.POI_BBOX
    m = folium.Map(location=[(S + N) / 2, (W + E) / 2],
                   zoom_start=15, tiles="OpenStreetMap")

    # POI_BBOX rectangle for spatial context
    folium.Rectangle(bounds=[(S, W), (N, E)], color="#000", weight=2,
                     fill=False, tooltip="POI_BBOX").add_to(m)

    accepted = folium.FeatureGroup(
        name="Accepted (trustworthy, reconciled)", show=True)
    disagree = folium.FeatureGroup(
        name="Disagree (g_dino & g_vlm > MAX_VAR_M, both plotted)",
        show=True)
    vlm_unres = folium.FeatureGroup(
        name="VLM unresolved (g_dino plotted; VLM guess not in OSM)",
        show=True)
    weak = folium.FeatureGroup(name="dino_weak (VLM-only GPS plotted)",
                               show=False)
    low = folium.FeatureGroup(name="legacy low_score (if present)",
                              show=False)

    cnt = collections.Counter()
    for r in rows:
        v = r["video"]
        col = vcolor[v]
        if r.get("accepted"):
            cnt["accepted"] += 1
            popup = folium.Popup(
                f"<b>{v}/{r['frame_id']}</b><br>"
                f"reconciled: <b>"
                f"{r['gps'][0]:.5f}, {r['gps'][1]:.5f}</b><br>"
                f"score: {r['score']:.3f} &nbsp;&middot;&nbsp; "
                f"variance: {r['variance_m']:.0f} m<br>"
                f"DINOv2 cos: {r['s_dino']:.3f} &rarr; "
                f"{r['g_dino'][0]:.5f}, {r['g_dino'][1]:.5f}<br>"
                f"VLM ({r['vlm_conf']}): {r['place_guess']} &rarr; "
                f"{r['g_vlm'][0]:.5f}, {r['g_vlm'][1]:.5f}<br>"
                f"heading: {r['heading']:.0f}&deg; "
                f"(spread {r['heading_spread']:.0f}&deg;)",
                max_width=380)
            folium.CircleMarker(
                location=r["gps"], radius=4, color=col,
                fill=True, fill_opacity=0.85, weight=1,
                tooltip=f"{v}/{r['frame_id']}", popup=popup,
            ).add_to(accepted)
        elif r.get("reject_reason") == "dino_weak" and r.get("g_vlm"):
            cnt["dino_weak"] += 1
            folium.CircleMarker(
                location=r["g_vlm"], radius=3, color="#888",
                fill=True, fill_opacity=0.35, weight=1,
                tooltip=(f"dino_weak: {v}/{r['frame_id']}  "
                         f"cos {r['s_dino']:.2f}")
            ).add_to(weak)
        elif r.get("reject_reason") == "disagree" and r.get("g_vlm"):
            cnt["disagree"] += 1
            popup = folium.Popup(
                f"<b>{v}/{r['frame_id']}</b> &mdash; <b>disagree</b><br>"
                f"variance: <b>{r.get('variance_m') or 0:.0f} m</b><br>"
                f"DINOv2 cos: {r['s_dino']:.3f} &rarr; "
                f"{r['g_dino'][0]:.5f}, {r['g_dino'][1]:.5f}"
                f" (nearest OSM: <i>{r.get('dino_nearest_name', '?')}</i>)<br>"
                f"VLM ({r['vlm_conf']}): <b>{r['place_guess']}</b> &rarr; "
                f"{r['g_vlm'][0]:.5f}, {r['g_vlm'][1]:.5f}<br>"
                f"&nbsp; <i>{r.get('reasoning', '')}</i>",
                max_width=380)
            # plot BOTH g_dino (blue) and g_vlm (orange) so the
            # disagreement is visible at a glance
            folium.CircleMarker(
                location=r["g_dino"], radius=3, color="#1f3a68",
                fill=True, fill_opacity=0.6, weight=1,
                tooltip=f"DINOv2 pin (disagree): {v}/{r['frame_id']}",
                popup=popup,
            ).add_to(disagree)
            folium.CircleMarker(
                location=r["g_vlm"], radius=3, color="#CC6600",
                fill=True, fill_opacity=0.6, weight=1,
                tooltip=f"VLM pin (disagree): {v}/{r['frame_id']}",
                popup=popup,
            ).add_to(disagree)
        elif r.get("reject_reason") == "vlm_unresolved" and r.get("g_dino"):
            cnt["vlm_unresolved"] += 1
            popup = folium.Popup(
                f"<b>{v}/{r['frame_id']}</b> &mdash; "
                f"<b>vlm_unresolved</b><br>"
                f"DINOv2 cos: {r['s_dino']:.3f} &rarr; "
                f"{r['g_dino'][0]:.5f}, {r['g_dino'][1]:.5f}<br>"
                f"VLM said: <b>'{r.get('vlm_guess_raw', '')}'</b> "
                f"(no OSM match)<br>"
                f"&nbsp; <i>{r.get('reasoning', '')}</i>",
                max_width=380)
            folium.CircleMarker(
                location=r["g_dino"], radius=5, color="#9933CC",
                fill=True, fill_opacity=0.8, weight=2,
                tooltip=(f"vlm_unresolved: {v}/{r['frame_id']}  "
                         f"VLM='{r.get('vlm_guess_raw', '')}'"),
                popup=popup,
            ).add_to(vlm_unres)
        elif r.get("reject_reason") == "low_score" and r.get("g_vlm"):
            cnt["low_score"] += 1
            folium.CircleMarker(
                location=r["g_vlm"], radius=3, color="#AA00AA",
                fill=True, fill_opacity=0.4, weight=1,
                tooltip=f"legacy low_score: {v}/{r['frame_id']}"
            ).add_to(low)
        else:
            cnt["other"] += 1

    accepted.add_to(m)
    disagree.add_to(m)
    vlm_unres.add_to(m)
    weak.add_to(m)
    low.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # legend (fixed HTML overlay)
    swatches = "".join(
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'background:{vcolor[v]};margin-right:4px;border-radius:50%;">'
        f'</span>{v}&nbsp;&nbsp;' for v in videos)
    legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; background: white;
         padding: 10px 12px; border: 1px solid #888; border-radius: 6px;
         font-family: Arial, sans-serif; font-size: 12px; z-index: 9999;
         box-shadow: 0 1px 4px rgba(0,0,0,0.2); max-width: 420px;">
      <b>GPS recovery (strict F1+F2+F3)</b><br>
      <b>Accepted ({cnt['accepted']})</b> &mdash; reconciled GPS, coloured
      by video. Click any dot for full detail:<br>{swatches}<br>
      <span style="color:#1f3a68;">&#9679;</span>
      <span style="color:#CC6600;">&#9679;</span>
      <b>Disagree ({cnt['disagree']})</b> &mdash; DINOv2 pin (blue) +
      VLM pin (orange) drawn together so you see the disagreement.<br>
      <span style="color:#9933CC;">&#9679;</span>
      <b>VLM unresolved ({cnt['vlm_unresolved']})</b> &mdash; DINOv2 has
      a real match, but the VLM's guess didn't resolve to any OSM POI
      (tooltip shows the raw VLM guess &mdash; often a real place
      missing from our OSM table).<br>
      <span style="color:#888;">&#9679;</span> dino_weak
      ({cnt['dino_weak']}, toggle on to view).
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    out = config.VIZ_DIR / "gps_recovery_map.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"wrote {out}")
    for k, v in cnt.items():
        print(f"  {k:12s} {v}")


if __name__ == "__main__":
    build()
