"""Annotation QA viz — "was the direction told correctly?".

Reads `annotations_<variant>.jsonl` and renders a Leaflet map for
human inspection. Per kept sample:

  · green dot     = the frame's GPS (camera position)
  · short arrow   = recovered camera heading (the direction the
                    walker is facing right now)
  · coloured line = OSM walking route to the destination
                    (green = verifier passed, red = failed)
  · pin           = destination POI
  · click popup   = the photo + the spoken answer + verifier δ +
                    the thinking trace

Use this to *audit* the dataset before training: scroll the map, click
20 random samples, check that the answer ("turn left at the tram
tracks") actually agrees with the green polyline. Visual QA catches
failure modes (e.g. systematic 180° heading flips on one video) that
the verifier δ alone misses.

  python -m src.viz_annotate                                 # default file
  python -m src.viz_annotate --input annotations_strict.jsonl
  python -m src.viz_annotate --sample 100 --seed 0           # random 100
  python -m src.viz_annotate --only-failed                   # δ>=30 only
"""

import argparse
import base64
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402


def _heading_endpoint(lat, lon, heading_deg, length_m=15.0):
    R = 6_371_000.0
    br = math.radians(heading_deg)
    dlat = (length_m * math.cos(br)) / R
    dlon = (length_m * math.sin(br)) / (R * math.cos(math.radians(lat)))
    return (lat + math.degrees(dlat), lon + math.degrees(dlon))


def _thumb_data_uri(image_path, max_px=320):
    """Small base64 JPEG so popups are self-contained (no file:// URLs
    that some browsers refuse to load from a Leaflet HTML)."""
    from PIL import Image
    import io
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=78)
    return ("data:image/jpeg;base64,"
            + base64.b64encode(buf.getvalue()).decode())


def _popup_html(rec, thumb_uri):
    """A self-contained popup combining photo + answer + diagnostics."""
    color = "green" if rec.get("accepted") else "#c33"
    verb = rec.get("action") or "(no verb parsed)"
    delta = rec.get("verifier_delta")
    delta_s = f"{delta:.0f}°" if delta is not None else "—"
    nearby = ", ".join(rec.get("nearby_pois", [])[:5]) or "—"
    answer = (rec.get("answer") or "").replace("<", "&lt;").replace(
        ">", "&gt;")
    thinking = (rec.get("thinking") or "").replace("<", "&lt;").replace(
        ">", "&gt;")[:600]
    return (
        f'<div style="font:13px/1.35 system-ui;max-width:340px">'
        f'<img src="{thumb_uri}" '
        f'style="width:320px;border:1px solid #888"><br>'
        f'<b>{rec["video"]}/{rec["frame_id"]}</b> → '
        f'<b>{rec["dest_name"]}</b><br>'
        f'<i>heading={rec["heading"]:.0f}° · '
        f'route_bearing={rec["route_bearing"]:.0f}° · '
        f'δ=<span style="color:{color}"><b>{delta_s}</b></span> · '
        f'verb=<b>{verb}</b></i><br>'
        f'<p style="margin:6px 0"><b>Answer:</b> {answer}</p>'
        f'<details><summary>thinking</summary>'
        f'<pre style="white-space:pre-wrap;font-size:11px">{thinking}'
        f'</pre></details>'
        f'<small>Nearby: {nearby}</small>'
        f'</div>'
    )


def build_map(records):
    import folium

    W, S, E, N = config.POI_BBOX
    m = folium.Map(location=[(S + N) / 2, (W + E) / 2], zoom_start=15,
                   tiles="cartodbpositron")
    folium.Rectangle(bounds=[[S, W], [N, E]],
                     color="#000", weight=1, fill=False).add_to(m)

    fg_pass = folium.FeatureGroup(name="passed (green)", show=True)
    fg_fail = folium.FeatureGroup(name="failed (red)", show=True)
    fg_arrow = folium.FeatureGroup(name="heading arrows", show=True)

    for rec in records:
        lat, lon = rec["gps"]
        dlat, dlon = rec["dest_gps"]
        polyline = rec.get("route_latlon") or [rec["gps"], rec["dest_gps"]]
        accepted = rec.get("accepted")
        line_color = "#2a9d8f" if accepted else "#e63946"

        img_path = (config.FRAMES_DIR / rec["video"] /
                    f"{rec['frame_id']}.jpg")
        try:
            thumb = _thumb_data_uri(img_path)
        except (FileNotFoundError, OSError):
            thumb = ""
        popup = folium.Popup(_popup_html(rec, thumb), max_width=360)

        fg = fg_pass if accepted else fg_fail

        folium.PolyLine(polyline, color=line_color, weight=4,
                        opacity=0.7).add_to(fg)
        folium.CircleMarker(
            [lat, lon], radius=6, color=line_color, fill=True,
            fill_color=line_color, fill_opacity=0.9, popup=popup,
        ).add_to(fg)
        folium.Marker(
            [dlat, dlon],
            icon=folium.Icon(color="green" if accepted else "red",
                             icon="flag"),
            tooltip=rec["dest_name"],
        ).add_to(fg)

        end = _heading_endpoint(lat, lon, rec["heading"], length_m=20.0)
        folium.PolyLine([[lat, lon], end], color="#3a86ff",
                        weight=3, opacity=0.9).add_to(fg_arrow)

    for f in (fg_pass, fg_fail, fg_arrow):
        f.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", default=None,
                    help="annotations*.jsonl (default: any in CITY_DIR)")
    ap.add_argument("--output", default=None,
                    help="HTML out (default viz/annotate_<stem>.html)")
    ap.add_argument("--sample", type=int, default=60,
                    help="random sample size; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-passed", action="store_true")
    ap.add_argument("--only-failed", action="store_true")
    args = ap.parse_args()

    if args.input:
        in_path = Path(args.input)
    else:
        glob = sorted(config.CITY_DIR.glob("annotations*.jsonl"))
        if not glob:
            sys.exit("[viz_annotate] no annotations*.jsonl found in "
                     f"{config.CITY_DIR}")
        in_path = glob[-1]
        print(f"[viz_annotate] using {in_path.name}", flush=True)

    records = [json.loads(l) for l in in_path.open(encoding="utf-8")
               if l.strip()]
    print(f"[viz_annotate] loaded {len(records)} records", flush=True)

    if args.only_passed:
        records = [r for r in records if r.get("accepted")]
    if args.only_failed:
        records = [r for r in records if not r.get("accepted")]

    rng = random.Random(args.seed)
    if args.sample and len(records) > args.sample:
        records = rng.sample(records, args.sample)

    verbs = Counter(r.get("action") for r in records)
    print(f"[viz_annotate] verbs: {dict(verbs)}", flush=True)

    m = build_map(records)
    out = (Path(args.output) if args.output
           else config.VIZ_DIR / f"annotate_{in_path.stem}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"[viz_annotate] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
