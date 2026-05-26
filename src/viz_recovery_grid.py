"""Per-frame HTML grid for GPS-recovery sanity checking.

For each frame shows:
  - original photo (QUERY, blue border)
  - top-K Street View matches by DINOv2 cosine
  - VLM info: raw guess, confidence, OSM resolution, reasoning
  - DINOv2 info: top SV pano, nearest OSM POI, distance
  - reconcile diagnostics: variance, semantic_match, spatial_match

Grouped into three sections: ACCEPTED / DISAGREE / VLM UNRESOLVED.

    python -m src.viz_recovery_grid                # 30 rows / section
    python -m src.viz_recovery_grid --limit 0      # all rows (~480)
    python -m src.viz_recovery_grid -k 5           # show top-5 SV
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                # noqa: E402


def _file_url(p):
    return "file:///" + quote(str(Path(p).resolve()).replace("\\", "/"))


def build():
    ap = argparse.ArgumentParser(
        description="Per-frame photo grid for GPS-recovery sanity check")
    ap.add_argument("--limit", type=int, default=30,
                    help="cap rows per section (0 = all)")
    ap.add_argument("--input", type=str, default="gps_recovery_all.jsonl",
                    help="input jsonl under data/cities/zurich/ "
                         "(default 'gps_recovery_all.jsonl'; pilot is "
                         "'gps_recovery.jsonl')")
    ap.add_argument("--output", type=str,
                    default="gps_recovery_all_grid.html",
                    help="output filename under viz/")
    ap.add_argument("--frame-cache", type=str, default="frames_n1_l0",
                    help="DINOv2 frame cache name (must match the "
                         "gps_recovery input)")
    ap.add_argument("--min-sim", type=float, default=0.0,
                    help="filter rows by s_dino >= this (default 0 = "
                         "no filter; the input is already filtered to "
                         "cos>=MIN_SIM at gps_recovery time)")
    ap.add_argument("--tier", type=int, choices=[0, 1, 2], default=0,
                    help="filter rows by tier (default 0 = no filter; "
                         "1 = VLM-confirmed only, the 'VLM-agreed' set "
                         "when combined with the ACCEPTED section; "
                         "2 = visual-match only)")
    ap.add_argument("--random", action="store_true",
                    help="random sample within each section instead of "
                         "taking the first --limit rows")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for --random (default 42)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (config.CITY_DIR / args.input).open(encoding="utf-8")
            if l.strip()]
    if not rows:
        sys.exit(f"no {args.input} — run `python -m src.gps_recovery`")

    # DINOv2 caches for top-K reconstruction (only top_sv_id is in jsonl)
    cdir = config.CITY_DIR / "dinov2"
    sv_cache = np.load(cdir / "sv_v1.npz", allow_pickle=True)
    sv_embs = sv_cache["embs"]
    sv_ids = [Path(p).stem for p in sv_cache["paths"]]

    fcache = np.load(cdir / f"{args.frame_cache}.npz", allow_pickle=True)
    frame_embs = fcache["embs"]
    frame_paths = [Path(p) for p in fcache["paths"]]
    frame_idx = {(p.parent.name, p.stem): i for i, p in enumerate(frame_paths)}

    sv_dir = config.STREETVIEW_DIR / "images"

    # SV meta — id -> {pano_id, compass_angle} — needed to group crops
    # by pano (so we can show all 4 compass crops at top-1's pano).
    sv_meta_path = config.STREETVIEW_DIR / "meta.jsonl"
    sv_meta = {}
    for line in sv_meta_path.open(encoding="utf-8"):
        m = json.loads(line)
        sv_meta[m["id"]] = {"pano_id": m.get("pano_id", ""),
                            "heading": m.get("compass_angle", 0)}
    pano_to_crops = collections.defaultdict(list)
    for j, sid in enumerate(sv_ids):
        pid = sv_meta.get(sid, {}).get("pano_id", "")
        if pid:
            pano_to_crops[pid].append(j)

    sections = [
        ("ACCEPTED", "accepted", "good",
         "Trustworthy frames — DINOv2 ≥ 0.6, VLM resolved, "
         "F3 (semantic OR ≤150 m) passed.",
         lambda r: r["accepted"]),
        ("DISAGREE", "disagree", "bad",
         "Both signals available but pointing to different places "
         "(>150 m apart AND OSM name mismatch).",
         lambda r: r.get("reject_reason") == "disagree"),
        ("VLM UNRESOLVED", "vlm_unresolved", "warn",
         "DINOv2 matched but the VLM's raw guess didn't resolve to any "
         "OSM POI — these names may be real places missing from our "
         "OSM table.",
         lambda r: r.get("reject_reason") == "vlm_unresolved"),
    ]

    def row_html(r):
        # query frame (the actual video photo)
        fpath = config.FRAMES_DIR / r["video"] / f"{r['frame_id']}.jpg"
        cells = [
            f'<div class="cell q"><img src="{_file_url(fpath)}" loading="lazy">'
            f'<div class="label"><b>QUERY</b><br>'
            f'{r["video"]}/{r["frame_id"]}<br>'
            f'heading: <b>{r.get("heading") or 0:.0f}&deg;</b></div></div>'
        ]
        # All 4 compass crops at top-1's PANO — the heading decision is
        # made from these. Sorted N→E→S→W; the highest-cosine crop is
        # highlighted in red (this is the direction DINOv2 picked).
        key = (r["video"], r["frame_id"])
        crops = []
        if key in frame_idx:
            sims_all = sv_embs @ frame_embs[frame_idx[key]]
            top_idx = int(np.argmax(sims_all))
            top_pano = sv_meta.get(sv_ids[top_idx], {}).get("pano_id", "")
            crop_js = pano_to_crops.get(top_pano, [])
            crops = sorted(
                ((sv_meta[sv_ids[j]]["heading"], float(sims_all[j]), j)
                 for j in crop_js),
                key=lambda t: t[0])
            best_cos = max((c for _, c, _ in crops), default=0.0)
            for h_deg, cos, j in crops:
                sv_path = sv_dir / f"{sv_ids[j]}.jpg"
                is_best = abs(cos - best_cos) < 1e-9
                cls = "good" if cos > 0.75 else ("ok" if cos >= 0.60 else "bad")
                star = " &#9733;" if is_best else ""
                # red outline on the chosen direction
                style = (' style="outline: 3px solid #d62728; '
                         'outline-offset: -3px;"' if is_best else '')
                cells.append(
                    f'<div class="cell">'
                    f'<img src="{_file_url(sv_path)}" loading="lazy"{style}>'
                    f'<div class="label">heading <b>{int(h_deg):03d}&deg;</b>'
                    f'{star}<br>'
                    f'<span class="{cls}">cos {cos:.3f}</span></div>'
                    f'</div>')

        guess = r.get("vlm_guess_raw", "")
        resolved = r.get("place_guess") or "&lt;not in OSM&gt;"
        nearest = r.get("dino_nearest_name") or "&mdash;"
        near_m = r.get("dino_nearest_m") or 0
        var_m = r.get("variance_m") or 0
        poi_d = r.get("poi_dist_m")
        exact = r.get("exact_name_match")
        nbhd = r.get("neighborhood_match")
        spa = r.get("spatial_match")
        heading = r.get("heading") or 0
        spread = r.get("heading_spread")
        hgap = r.get("heading_gap")
        npano = r.get("same_pano_heading_count") or 0

        # F3 outcome label
        if r.get("accepted"):
            if exact:
                f3 = '<span class="good">F3 pass &mdash; exact name match</span>'
            elif nbhd:
                f3 = (f'<span class="good">F3 pass &mdash; neighborhood'
                      f' (POI-POI {poi_d:.0f} m)</span>')
            else:
                f3 = '<span class="good">F3 pass</span>'
        elif r.get("reject_reason") == "disagree":
            poi_str = (f' (POI-POI {poi_d:.0f} m'
                       f' &gt; {config.NEIGHBORHOOD_RADIUS_M:.0f})'
                       if poi_d is not None else '')
            f3 = (f'<span class="bad">F3 fail &mdash; '
                  f'disagree{poi_str}</span>')
        else:
            f3 = (f'<span class="bad">{r.get("reject_reason", "")}'
                  f'</span>')

        # heading-gap colour
        if hgap is None:
            hg_cls = ""
        elif hgap >= 0.15:
            hg_cls = "good"
        elif hgap >= 0.05:
            hg_cls = "ok"
        else:
            hg_cls = "bad"

        # ── heading calculation, showing the math for this frame ──
        if crops:
            sum_sin = sum(cos * math.sin(math.radians(h))
                          for h, cos, _ in crops)
            sum_cos = sum(cos * math.cos(math.radians(h))
                          for h, cos, _ in crops)
            h_calc = (math.degrees(math.atan2(sum_sin, sum_cos)) + 360) % 360
            terms = "&nbsp;+&nbsp;".join(
                f"{cos:.2f}&middot;<i>e<sup>i{int(h):03d}&deg;</sup></i>"
                for h, cos, _ in crops)
            heading_block = (
                f'<div class="calc"><b>heading calc</b> &mdash; '
                f'atan2(&Sigma; w&middot;sin&theta;, &Sigma; w&middot;cos&theta;) '
                f'over the 4 crops above:<br>'
                f'&nbsp;&nbsp;weights {terms}<br>'
                f'&nbsp;&nbsp;&Sigma; w&middot;sin&theta; = {sum_sin:+.3f}'
                f' &nbsp;&middot;&nbsp; &Sigma; w&middot;cos&theta; = '
                f'{sum_cos:+.3f}<br>'
                f'&nbsp;&nbsp;heading = atan2({sum_sin:+.3f}, '
                f'{sum_cos:+.3f}) = <b>{h_calc:.0f}&deg;</b></div>'
            )
        else:
            heading_block = ''

        info = (
            f'<div class="info">'
            f'<div><b>VLM guess:</b> "{guess}" &nbsp;'
            f'(<b>{r.get("vlm_conf", "?")}</b> confidence)</div>'
            f'<div><b>VLM resolved to:</b> {resolved}</div>'
            f'<div><b>DINOv2 top SV nearest OSM:</b> {nearest}'
            f' &nbsp;(SV pano is {near_m:.0f} m from it)</div>'
            f'<div><b>POI-to-POI distance (VLM &harr; DINOv2 nearest):</b>'
            f' {("&mdash;" if poi_d is None else f"{poi_d:.0f} m")}'
            f' &nbsp; threshold {config.NEIGHBORHOOD_RADIUS_M:.0f} m</div>'
            f'<div><b>variance |g_dino &minus; g_vlm|:</b> {var_m:.0f} m'
            f' &nbsp; <b>spatial (&le;150 m):</b> {spa}</div>'
            f'<div><b>heading:</b> {heading:.0f}&deg;'
            f' &nbsp; <b>same-pano gap:</b> '
            f'<span class="{hg_cls}">'
            f'{(hgap if hgap is None else f"{hgap:.2f}")}</span>'
            f' (of {npano} crops)</div>'
            f'{heading_block}'
            f'<div>{f3}</div>'
            f'<div class="reason"><i>{r.get("reasoning", "")}</i></div>'
            f'</div>'
        )
        return '<div class="row">' + ''.join(cells) + info + '</div>'

    body = []
    counts = {}
    rng = None
    if args.random:
        import random as _rand
        rng = _rand.Random(args.seed)
    for title, anchor, cls, desc, predicate in sections:
        matching = [r for r in rows if predicate(r)
                    and r.get("s_dino", 0) >= args.min_sim
                    and (args.tier == 0
                         or r.get("tier") == args.tier)]
        total = len(matching)
        if rng is not None:
            matching = matching[:]      # copy before shuffling
            rng.shuffle(matching)
        shown = matching[:args.limit] if args.limit else matching
        counts[title] = (len(shown), total)
        body.append(
            f'<div id="{anchor}" class="divider {cls}">{title} '
            f'&mdash; showing {len(shown)} of {total}'
            f'<div class="desc">{desc}</div></div>')
        body.extend(row_html(r) for r in shown)

    summary_line = " &nbsp;|&nbsp; ".join(
        f"{t}: <b>{counts[t][0]}/{counts[t][1]}</b>"
        for t, _, _, _, _ in sections)

    toc = " &nbsp;&middot;&nbsp; ".join(
        f'<a href="#{anchor}" class="{cls}">jump to {title}</a>'
        for title, anchor, cls, _, _ in sections)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>NavLM v2 - GPS recovery sanity check</title>
<style>
body {{ font-family: Arial, sans-serif; background: #f5f5f5;
        margin: 16px; }}
h1 {{ color: #1a1a1a; }}
.intro {{ background: #e9edf4; padding: 10px 14px; border-radius: 6px;
          margin-bottom: 16px; font-size: 13px; }}
.divider {{ padding: 12px 14px; color: white; font-weight: bold;
            margin: 22px 0 10px 0; border-radius: 6px; font-size: 15px; }}
.divider .desc {{ font-weight: normal; font-size: 12px; margin-top: 4px;
                  opacity: 0.95; }}
.divider.good {{ background: #2d7d2d; }}
.divider.bad  {{ background: #cc3333; }}
.divider.warn {{ background: #9933cc; }}
.row {{ display: flex; gap: 8px; margin-bottom: 10px; background: white;
        padding: 8px; border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        align-items: flex-start; }}
.cell {{ text-align: center; flex-shrink: 0; }}
.cell img {{ width: 210px; height: auto; border-radius: 4px; }}
.cell.q img {{ border: 3px solid #1f3a68; }}
.cell:not(.q) img {{ border: 2px solid #ccc; }}
.label {{ font-size: 11px; color: #555; margin-top: 4px; }}
.info {{ font-size: 12px; color: #333; flex-grow: 1;
         padding: 4px 10px; max-width: 520px; line-height: 1.5; }}
.info b {{ color: #1a1a1a; }}
.info .calc {{ background: #f0f4fa; border-left: 3px solid #1f3a68;
               padding: 4px 8px; margin: 6px 0; font-family:
               "Consolas", monospace; font-size: 11px; line-height: 1.5; }}
.info .reason {{ margin-top: 4px; color: #666; }}
.good {{ color: #2d7d2d; font-weight: bold; }}
.ok   {{ color: #b58900; }}
.bad  {{ color: #cc3333; }}
</style></head><body>
<h1>GPS recovery sanity check</h1>
<div class="intro">
{summary_line} &nbsp;|&nbsp; right side: <b>all 4 compass crops at
top-1's pano</b>, sorted N&rarr;E&rarr;S&rarr;W; the highest-cosine
crop is outlined in red &mdash; that's the direction the per-frame
heading is computed from.<br>
Cosine colour:
<span class="good">&gt;0.75 strong</span> /
<span class="ok">&ge;0.60 matched</span> /
<span class="bad">&lt;0.60 weak</span>
<br><br>
<b>Heading formula</b> (cosine-weighted circular mean of the 4
compass crops):
&nbsp;<code>heading = atan2( &Sigma; cos<sub>i</sub>&middot;sin&theta;<sub>i</sub>,
 &nbsp; &Sigma; cos<sub>i</sub>&middot;cos&theta;<sub>i</sub> )</code>
&nbsp; where &theta;<sub>i</sub>&in;&#123;0&deg;,90&deg;,180&deg;,270&deg;&#125;
and cos<sub>i</sub> is the cosine similarity of the query against that crop.
Each row's info panel shows the worked numbers for that frame.
<br><br><b>Jump:</b> {toc}
</div>
{''.join(body)}
</body></html>"""
    out = config.VIZ_DIR / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    for t, _, _, _, _ in sections:
        s, n = counts[t]
        print(f"  {t:18s}  shown {s} of {n}")


if __name__ == "__main__":
    build()
