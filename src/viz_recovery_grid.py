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
import json
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                # noqa: E402
from src.gps_recovery import cosine_topk     # noqa: E402


def _file_url(p):
    return "file:///" + quote(str(Path(p).resolve()).replace("\\", "/"))


def build():
    ap = argparse.ArgumentParser(
        description="Per-frame photo grid for GPS-recovery sanity check")
    ap.add_argument("--limit", type=int, default=30,
                    help="cap rows per section (0 = all)")
    ap.add_argument("-k", type=int, default=3,
                    help="show top-K SV matches per frame")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (config.CITY_DIR / "gps_recovery.jsonl").open(encoding="utf-8")
            if l.strip()]
    if not rows:
        sys.exit("no gps_recovery.jsonl — run `python -m src.gps_recovery`")

    # DINOv2 caches for top-K reconstruction (only top_sv_id is in jsonl)
    cdir = config.CITY_DIR / "dinov2"
    sv_cache = np.load(cdir / "sv_v1.npz", allow_pickle=True)
    sv_embs = sv_cache["embs"]
    sv_ids = [Path(p).stem for p in sv_cache["paths"]]

    fcache = np.load(cdir / "frames_n30_l0.npz", allow_pickle=True)
    frame_embs = fcache["embs"]
    frame_paths = [Path(p) for p in fcache["paths"]]
    frame_idx = {(p.parent.name, p.stem): i for i, p in enumerate(frame_paths)}

    sv_dir = config.STREETVIEW_DIR / "images"

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
            f'heading: {r.get("heading") or 0:.0f}&deg;</div></div>'
        ]
        # top-K SV matches recomputed from the cached embeddings
        key = (r["video"], r["frame_id"])
        if key in frame_idx:
            idx, sims = cosine_topk(
                frame_embs[frame_idx[key]], sv_embs, k=args.k)
            for j, s in zip(idx, sims):
                sv_id = sv_ids[int(j)]
                sv_path = sv_dir / f"{sv_id}.jpg"
                cls = "good" if s > 0.75 else ("ok" if s >= 0.60 else "bad")
                cells.append(
                    f'<div class="cell">'
                    f'<img src="{_file_url(sv_path)}" loading="lazy">'
                    f'<div class="label">{sv_id[:24]}<br>'
                    f'<span class="{cls}">cos {float(s):.3f}</span></div>'
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
            f'<div><b>heading:</b> {heading:.0f}&deg; '
            f' &nbsp; <b>spread (top-K):</b> '
            f'{(spread if spread is None else f"{spread:.0f}") }&deg;'
            f' &nbsp; <b>same-pano gap:</b> '
            f'<span class="{hg_cls}">'
            f'{(hgap if hgap is None else f"{hgap:.2f}")}</span>'
            f' (of {npano} crops)</div>'
            f'<div>{f3}</div>'
            f'<div class="reason"><i>{r.get("reasoning", "")}</i></div>'
            f'</div>'
        )
        return '<div class="row">' + ''.join(cells) + info + '</div>'

    body = []
    counts = {}
    for title, anchor, cls, desc, predicate in sections:
        matching = [r for r in rows if predicate(r)]
        total = len(matching)
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
.info .reason {{ margin-top: 4px; color: #666; }}
.good {{ color: #2d7d2d; font-weight: bold; }}
.ok   {{ color: #b58900; }}
.bad  {{ color: #cc3333; }}
</style></head><body>
<h1>GPS recovery sanity check</h1>
<div class="intro">
{summary_line} &nbsp;|&nbsp; top-{args.k} SV per frame
&middot; cosine colour:
<span class="good">&gt;0.75 strong</span> /
<span class="ok">&ge;0.60 matched</span> /
<span class="bad">&lt;0.60 weak</span>
<br><br><b>Jump:</b> {toc}
</div>
{''.join(body)}
</body></html>"""
    out = config.VIZ_DIR / "gps_recovery_grid.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    for t, _, _, _, _ in sections:
        s, n = counts[t]
        print(f"  {t:18s}  shown {s} of {n}")


if __name__ == "__main__":
    build()
