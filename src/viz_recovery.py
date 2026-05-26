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
    import argparse
    import folium
    ap = argparse.ArgumentParser(description="GPS recovery folium map")
    ap.add_argument("--input", type=str, default="gps_recovery_all.jsonl",
                    help="input jsonl under data/cities/zurich/ "
                         "(default 'gps_recovery_all.jsonl'; use "
                         "'gps_recovery.jsonl' for the pilot's 872-frame set)")
    ap.add_argument("--output", type=str,
                    default="gps_recovery_all_map.html",
                    help="output filename under viz/")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (config.CITY_DIR / args.input).open(encoding="utf-8")
            if l.strip()]
    if not rows:
        sys.exit(f"no {args.input} — run `python -m src.gps_recovery`")

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
    # g_vlm = VLM's resolved POI centroid (any frame with a vlm match).
    # Two sub-layers so each can be toggled independently. The accepted
    # frames' centroids are mostly near g_dino; the disagree frames'
    # centroids fly far (incl. Limmat / Zürichsee → 14 km off-bbox).
    vlm_cen_acc = folium.FeatureGroup(
        name="VLM centroid (g_vlm) — accepted", show=False)
    vlm_cen_dis = folium.FeatureGroup(
        name="VLM centroid (g_vlm) — disagree", show=False)

    cnt = collections.Counter()
    for r in rows:
        v = r["video"]
        col = vcolor[v]

        # VLM-resolved POI centroid (g_vlm). Plot on toggle layers so
        # the long-feature outliers (Limmat / Zürichsee centroids 14 km
        # off-bbox) are visible only when you ask for them.
        g_vlm = r.get("g_vlm")
        if g_vlm:
            d_to_vlm = r.get("variance_m")
            popup = folium.Popup(
                f"<b>g_vlm</b> for {v}/{r['frame_id']}<br>"
                f"VLM said: <b>{r.get('place_guess', '')}</b>"
                f" (resolved centroid)<br>"
                f"{('%.0f m' % d_to_vlm) if d_to_vlm is not None else '?'}"
                f" from g_dino", max_width=320)
            target = vlm_cen_acc if r.get("accepted") else (
                vlm_cen_dis if r.get("reject_reason") == "disagree"
                else None)
            if target is not None:
                folium.CircleMarker(
                    location=g_vlm, radius=3, color="#FF9900",
                    fill=True, fill_opacity=0.55, weight=1,
                    tooltip=(f"g_vlm: {r.get('place_guess', '')}  "
                             f"({d_to_vlm:.0f} m off)"
                             if d_to_vlm is not None else
                             f"g_vlm: {r.get('place_guess', '')}"),
                    popup=popup,
                ).add_to(target)

        if r.get("accepted"):
            cnt["accepted"] += 1
            # Tier-aware popup: tier-2 has no VLM data (g_vlm, variance,
            # place_guess all None). Render a smaller, simpler popup for
            # tier-2 to avoid TypeError on None formatting.
            heading_s = f"{(r.get('heading') or 0):.0f}"
            spread_s  = f"{(r.get('heading_spread') or 0):.0f}"
            if r.get("tier") == 2:
                popup_html = (
                    f"<b>{v}/{r['frame_id']}</b> &nbsp;"
                    f"<span style='color:#888;'>(tier 2 &mdash; "
                    f"DINOv2 only)</span><br>"
                    f"gps: <b>{r['gps'][0]:.5f}, {r['gps'][1]:.5f}</b>"
                    f" (= g_dino)<br>"
                    f"DINOv2 cos: <b>{r['s_dino']:.3f}</b><br>"
                    f"heading: {heading_s}&deg; &nbsp; "
                    f"<b>heading_gap:</b> "
                    f"{(r.get('heading_gap') or 0):.2f}<br>"
                    f"nearest OSM: <i>"
                    f"{r.get('dino_nearest_name') or '-'}</i> "
                    f"@ {(r.get('dino_nearest_m') or 0):.0f} m"
                )
            else:    # tier 1
                popup_html = (
                    f"<b>{v}/{r['frame_id']}</b> &nbsp;"
                    f"<span style='color:#1f3a68;'>(tier 1 &mdash; "
                    f"DINOv2 + VLM)</span><br>"
                    f"gps: <b>{r['gps'][0]:.5f}, "
                    f"{r['gps'][1]:.5f}</b><br>"
                    f"score: {(r.get('score') or 0):.3f} "
                    f"&middot; variance: "
                    f"{(r.get('variance_m') or 0):.0f} m<br>"
                    f"DINOv2 cos: {r['s_dino']:.3f} &rarr; "
                    f"{r['g_dino'][0]:.5f}, "
                    f"{r['g_dino'][1]:.5f}<br>"
                    f"VLM ({r.get('vlm_conf', '?')}): "
                    f"{r.get('place_guess', '-')}<br>"
                    f"heading: {heading_s}&deg; "
                    f"(spread {spread_s}&deg;)"
                )
            popup = folium.Popup(popup_html, max_width=380)
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
    vlm_cen_acc.add_to(m)
    vlm_cen_dis.add_to(m)
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
      <b>Accepted ({cnt['accepted']})</b> &mdash; g_dino (SV pano coords),
      coloured
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
      ({cnt['dino_weak']}, toggle on to view).<br>
      <span style="color:#FF9900;">&#9679;</span> <b>g_vlm centroids</b>
      &mdash; toggle on to see, per frame, where the VLM's resolved POI
      sits. For small POIs the centroid is close to <code>g_dino</code>;
      for long features (Limmat, Zürichsee) the centroid is
      <b>kilometres off</b> &mdash; exactly why we trust <code>g_dino</code>
      as the accepted GPS instead of blending in the centroid.
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    out = config.VIZ_DIR / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"wrote {out}")
    for k, v in cnt.items():
        print(f"  {k:12s} {v}")


if __name__ == "__main__":
    build()
