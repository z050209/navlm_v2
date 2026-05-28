"""Per-frame heading-direction sanity check, photo + map on one page.

Layout: split pane.
  Left  ─ interactive Leaflet map with every selected frame plotted
          as a marker + a heading arrow (the recovered direction the
          camera is facing). Pan/zoom freely.
  Right ─ scrollable photo gallery. Each card is one frame:
            * the frame's photo
            * a compass widget showing the recovered heading
            * frame_id / video / place_guess / GPS / heading number
          Click a card to fly the map to that frame and flash its
          marker — quick way to spot-check "does the heading number
          actually match what the camera shows?".

Use this when you want to scroll a sample of trusted_frames and ask
of each row: *is the heading reasonable for this photo?* A frame
where DINOv2 picked the wrong direction at a symmetric facade will
make this obvious — the photo shows N, the compass needle points S.

  python -m src.viz_heading_check                       # default 30 random
  python -m src.viz_heading_check --n 60 --seed 7
  python -m src.viz_heading_check --input trusted_frames.jsonl
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402

VIDEO_COLORS = {
    "bahnhofstrasse":   "#e6194B",
    "hidden_streets":   "#3cb44b",
    "looks_perfect":    "#cc9e00",
    "most_elegant":     "#4363d8",
    "most_famous":      "#f58231",
    "old_town_limmat":  "#911eb4",
    "zurich_main":      "#1899b8",
    "saturday_morning": "#000000",
}


def _thumb_data_uri(image_path, max_px=320):
    """Small JPEG base64 so the HTML is self-contained (no
    file:/// hash issues on some browsers)."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=78)
    return ("data:image/jpeg;base64,"
            + base64.b64encode(buf.getvalue()).decode())


def _load_rows(in_path, n, seed):
    rows = [json.loads(l) for l in in_path.open(encoding="utf-8")
            if l.strip()]
    rows = [r for r in rows if r.get("gps") and r.get("heading") is not None]
    rng = random.Random(seed)
    if n and n < len(rows):
        rows = rng.sample(rows, n)
    rows.sort(key=lambda r: (r["video"], r["frame_id"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "trusted_frames.jsonl"))
    ap.add_argument("--output",
                    default=str(config.VIZ_DIR / "heading_check.html"))
    ap.add_argument("--n", type=int, default=30,
                    help="how many frames to render (random sample)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def _resolve(p):
        path = Path(p)
        if path.exists() or path.is_absolute():
            return path
        return config.CITY_DIR / path.name

    in_path = _resolve(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(in_path, args.n, args.seed)
    print(f"[viz_heading_check] in: {in_path.name}  -> rendering "
          f"{len(rows)} frames", flush=True)

    # Centre the map on the geographic centre of the cohort
    if rows:
        clat = sum(r["gps"][0] for r in rows) / len(rows)
        clon = sum(r["gps"][1] for r in rows) / len(rows)
    else:
        clat, clon = 47.374, 8.541

    # Build the per-frame JS records (with the base64 photo).
    frame_js = []
    for r in rows:
        img_path = (config.FRAMES_DIR / r["video"] /
                    f"{r['frame_id']}.jpg")
        try:
            photo = _thumb_data_uri(img_path)
        except (FileNotFoundError, OSError):
            photo = ""
        frame_js.append({
            "video": r["video"], "frame_id": r["frame_id"],
            "lat": r["gps"][0], "lon": r["gps"][1],
            "heading": float(r["heading"]),
            "heading_gap": float(r.get("heading_gap") or 0.0),
            "place_guess": r.get("place_guess", ""),
            "segment_bearing": (None
                                 if r.get("segment_bearing") is None
                                 else float(r["segment_bearing"])),
            "color": VIDEO_COLORS.get(r["video"], "#666"),
            "photo": photo,
        })

    frames_json = json.dumps(frame_js, ensure_ascii=False)
    POI_BBOX = config.POI_BBOX

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>heading-check — {len(rows)} frames</title>
<link rel="stylesheet"
  href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body {{ margin:0; padding:0; height:100%; font:13px/1.35 system-ui; }}
  #wrap {{ display:flex; height:100vh; }}
  #map  {{ flex:1.4; background:#eee; }}
  #gallery {{ flex:1; overflow-y:scroll; padding:8px; background:#f6f6f6;
              border-left:1px solid #ccc; }}
  #header {{ position:fixed; top:10px; left:50px; z-index:9999;
             background:white; border:1px solid #888; padding:8px 12px;
             font:13px/1.4 system-ui; max-width:360px; }}
  .card {{ display:flex; gap:10px; background:white; border:1px solid #ddd;
           border-radius:4px; padding:8px; margin-bottom:10px;
           cursor:pointer; transition:box-shadow .15s; }}
  .card:hover {{ box-shadow:0 0 0 2px #2a9d8f; }}
  .card.active {{ box-shadow:0 0 0 3px #e76f51; }}
  .card img {{ width:220px; height:auto; max-height:160px;
               object-fit:cover; border:1px solid #888; }}
  .card .info {{ flex:1; }}
  .card .info b {{ color:#222; }}
  .card .info .kv {{ color:#555; font-size:12px; }}
  .compass {{ width:64px; height:64px; flex:0 0 64px; }}
  .swatch {{ display:inline-block; width:10px; height:10px;
             vertical-align:middle; margin-right:4px; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="map"></div>
  <div id="gallery"></div>
</div>
<div id="header">
  <b>Heading-direction sanity check</b> · {len(rows)} frames<br>
  <i>Left</i>: map with recovered heading arrows (one per frame, in the
  video's colour).<br>
  <i>Right</i>: photo + compass per frame. <b>Click a card</b> to fly
  the map to that frame.<br>
  <i>What to check</i>: does the compass needle direction look
  plausible for what the photo shows?
</div>

<script>
const FRAMES = {frames_json};

const map = L.map('map').setView([{clat}, {clon}], 16);
L.tileLayer(
  'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom:19,
    attribution:'&copy; OpenStreetMap contributors'}}).addTo(map);
L.rectangle(
  [[{POI_BBOX[1]}, {POI_BBOX[0]}], [{POI_BBOX[3]}, {POI_BBOX[2]}]],
  {{color:'#000', weight:1, fill:false}}).addTo(map);

// helper: a short polyline from (lat,lon) in `bearing` direction
function arrowEnd(lat, lon, bearing, lengthM) {{
  const R = 6371000;
  const br = bearing * Math.PI / 180;
  const dlat = (lengthM * Math.cos(br)) / R;
  const dlon = (lengthM * Math.sin(br)) /
                (R * Math.cos(lat * Math.PI / 180));
  return [lat + dlat * 180/Math.PI, lon + dlon * 180/Math.PI];
}}

const markers = [];
FRAMES.forEach((f, i) => {{
  const dot = L.circleMarker([f.lat, f.lon], {{
    radius:5, color:f.color, fillColor:f.color,
    fillOpacity:0.9, weight:1.5,
  }}).bindTooltip(`${{f.video}}/${{f.frame_id}}<br>heading ${{f.heading.toFixed(0)}}°`)
    .addTo(map);
  const end = arrowEnd(f.lat, f.lon, f.heading, 25);
  L.polyline([[f.lat, f.lon], end],
    {{color:f.color, weight:3, opacity:0.95}}).addTo(map);
  // little arrowhead — a thicker dot at the tip
  L.circleMarker(end, {{
    radius:3, color:f.color, fillColor:f.color,
    fillOpacity:1, weight:0,
  }}).addTo(map);
  markers.push(dot);
}});

// gallery cards
const gallery = document.getElementById('gallery');
let activeIdx = -1;
function activate(i) {{
  if (activeIdx >= 0) {{
    document.querySelectorAll('.card')[activeIdx].classList.remove('active');
  }}
  activeIdx = i;
  const card = document.querySelectorAll('.card')[i];
  card.classList.add('active');
  const f = FRAMES[i];
  map.flyTo([f.lat, f.lon], 18, {{duration:0.6}});
  markers[i].openTooltip();
}}
FRAMES.forEach((f, i) => {{
  const card = document.createElement('div');
  card.className = 'card';
  card.onclick = () => activate(i);
  const sw = `<span class="swatch" style="background:${{f.color}}"></span>`;
  const heading_x = (24 * Math.sin(f.heading * Math.PI / 180)).toFixed(2);
  const heading_y = (-24 * Math.cos(f.heading * Math.PI / 180)).toFixed(2);
  card.innerHTML = `
    <img src="${{f.photo}}" alt="frame">
    <div class="info">
      <b>${{sw}}${{f.video}}/${{f.frame_id}}</b><br>
      <span class="kv">place_guess: <b>${{f.place_guess}}</b></span><br>
      <span class="kv">heading: <b>${{f.heading.toFixed(0)}}°</b>
        &nbsp;heading_gap ${{f.heading_gap.toFixed(2)}}</span><br>
      <span class="kv">GPS: ${{f.lat.toFixed(5)}}, ${{f.lon.toFixed(5)}}</span><br>
      ${{f.segment_bearing !== null
         ? `<span class="kv">HMM edge bearing: ${{f.segment_bearing.toFixed(0)}}°</span>`
         : '' }}
    </div>
    <svg class="compass" viewBox="-32 -32 64 64">
      <circle cx="0" cy="0" r="28" fill="white" stroke="#888"/>
      <text x="0" y="-19" text-anchor="middle" font-size="9"
            fill="#444">N</text>
      <text x="19" y="3" text-anchor="middle" font-size="8"
            fill="#aaa">E</text>
      <text x="0" y="24" text-anchor="middle" font-size="8"
            fill="#aaa">S</text>
      <text x="-19" y="3" text-anchor="middle" font-size="8"
            fill="#aaa">W</text>
      <line x1="0" y1="0" x2="${{heading_x}}" y2="${{heading_y}}"
            stroke="${{f.color}}" stroke-width="2.5" stroke-linecap="round"/>
      <circle cx="${{heading_x}}" cy="${{heading_y}}" r="2.5"
              fill="${{f.color}}"/>
      <circle cx="0" cy="0" r="2" fill="#222"/>
    </svg>
  `;
  gallery.appendChild(card);
}});
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[viz_heading_check] wrote {out_path}  "
          f"({out_path.stat().st_size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
