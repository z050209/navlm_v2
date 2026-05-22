"""Stage 3 — Street View reference grid.

Builds the reference image index that video frames are matched against.
Two steps:

  scan      FREE metadata scan over a grid — finds every panorama ID +
            exact GPS. Writes panos.jsonl. $0.
  download  Street View Static API ($7/1000) — downloads SV_HEADINGS
            crops per panorama. Writes images/ + meta.jsonl.

Crawl bbox = bounding box of the candidate POIs the videos visit
(src/poi.py) + config.SV_MARGIN_M margin (DEV_MANUAL §2.4, Q3).

    python -m src.streetview --bbox                # print derived bbox
    python -m src.streetview --scan                # FREE
    python -m src.streetview --download [--max-panos N]   # costs $

Needs GOOGLE_MAPS_API_KEY (copy .env.example -> .env).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config            # noqa: E402
from src import poi      # noqa: E402

LAT_M = 111_320.0
SV_META = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMG = "https://maps.googleapis.com/maps/api/streetview"


def bbox_from_pois(pois, margin_m: float = config.SV_MARGIN_M):
    """(W, S, E, N) bounding box around POI coords + margin. Pure.
    `pois`: iterable of (en, zh, lat, lon, kind) tuples or (lat, lon)."""
    lats, lons = [], []
    for p in pois:
        if len(p) >= 5:      # CANDIDATE_POIS row
            lats.append(p[2]); lons.append(p[3])
        else:                # (lat, lon)
            lats.append(p[0]); lons.append(p[1])
    mid = sum(lats) / len(lats)
    dlat = margin_m / LAT_M
    dlon = margin_m / (LAT_M * math.cos(math.radians(mid)))
    return (min(lons) - dlon, min(lats) - dlat,
            max(lons) + dlon, max(lats) + dlat)


def grid_points(bbox, spacing_m: float = config.SV_GRID_M):
    """Grid of (lat, lon) covering bbox at ~spacing_m. Pure."""
    w, s, e, n = bbox
    lon_m = LAT_M * math.cos(math.radians((s + n) / 2))
    nx = max(1, round((e - w) * lon_m / spacing_m))
    ny = max(1, round((n - s) * LAT_M / spacing_m))
    return [(s + (n - s) * j / ny, w + (e - w) * i / nx)
            for i in range(nx + 1) for j in range(ny + 1)]


def bbox_from_scan(scan_path=None, pois_path=None,
                   margin_m=config.SV_MARGIN_M):
    """Crawl bbox from the POI scan — the POIs the videos *actually*
    visit (`poi_scan.jsonl` `matched`), looked up in `pois.json`, +
    margin. Returns (W, S, E, N). Raises ValueError if no matches."""
    scan_path = Path(scan_path or (config.CITY_DIR / "poi_scan.jsonl"))
    pois_path = Path(pois_path or (config.CITY_DIR / "pois.json"))
    gps = {p["name"]: (p["lat"], p["lon"])
           for p in json.loads(pois_path.read_text(encoding="utf-8"))}
    coords = []
    for ln in scan_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        for m in json.loads(ln).get("matched", []):
            g = gps.get(m.get("osm_name"))
            if g:
                coords.append(g)
    if not coords:
        raise ValueError("no matched POIs in the scan")
    return bbox_from_pois(coords, margin_m)


def crawl_bbox():
    """The derived crawl bbox + margin (Q3). Prefers the POI scan (the
    POIs the videos actually visit); falls back to the 27 candidates."""
    if (config.CITY_DIR / "poi_scan.jsonl").exists():
        try:
            return bbox_from_scan()
        except ValueError:
            pass
    return bbox_from_pois(poi.CANDIDATE_POIS)


def _api_key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not key:
        sys.exit("set GOOGLE_MAPS_API_KEY (copy .env.example -> .env)")
    return key


def scan():
    """FREE metadata scan over the grid -> panos.jsonl (deduped)."""
    import requests
    key = _api_key()
    pts = grid_points(crawl_bbox())
    config.STREETVIEW_DIR.mkdir(parents=True, exist_ok=True)
    panos_path = config.STREETVIEW_DIR / "panos.jsonl"

    seen = set()
    if panos_path.exists():
        seen = {json.loads(l)["pano_id"]
                for l in panos_path.open(encoding="utf-8")}
    print(f"[scan] {len(pts)} grid points, {len(seen)} panos already known")

    n_new = 0
    with panos_path.open("a", encoding="utf-8") as f:
        for lat, lon in tqdm(pts, desc="[scan]", unit="pt"):
            r = requests.get(SV_META, params={
                "location": f"{lat},{lon}", "key": key, "source": "outdoor",
            }, timeout=30).json()
            if r.get("status") != "OK":
                continue
            pid = r.get("pano_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            loc = r.get("location", {})
            f.write(json.dumps({"pano_id": pid, "lat": loc.get("lat"),
                                "lon": loc.get("lng"),
                                "date": r.get("date", "")}) + "\n")
            n_new += 1
    print(f"[scan] done — {len(seen)} unique panos ({n_new} new), "
          f"$0 -> {panos_path}")


def download(max_panos: int = 0):
    """Street View Static API -> images/ + meta.jsonl ($7/1000)."""
    import requests
    key = _api_key()
    panos_path = config.STREETVIEW_DIR / "panos.jsonl"
    if not panos_path.exists():
        sys.exit("run `python -m src.streetview --scan` first")
    panos = [json.loads(l) for l in panos_path.open(encoding="utf-8")]
    if max_panos:
        panos = panos[:max_panos]

    img_dir = config.STREETVIEW_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = config.STREETVIEW_DIR / "meta.jsonl"
    done = set()
    if meta_path.exists():
        done = {json.loads(l)["id"] for l in meta_path.open(encoding="utf-8")}

    n_target = len(panos) * len(config.SV_HEADINGS)
    print(f"[download] {len(panos)} panos x {len(config.SV_HEADINGS)} "
          f"headings = {n_target} images  (~${n_target * 0.007:.2f})")

    n_new = 0
    with meta_path.open("a", encoding="utf-8") as f:
        for p in tqdm(panos, desc="[download]", unit="pano"):
            for h in config.SV_HEADINGS:
                img_id = f"{p['pano_id']}_h{h:03d}"
                if img_id in done:
                    continue
                r = requests.get(SV_IMG, params={
                    "size": config.SV_IMG_SIZE, "pano": p["pano_id"],
                    "heading": h, "fov": config.SV_FOV, "pitch": 0,
                    "key": key,
                }, timeout=60)
                if r.status_code != 200:
                    continue
                (img_dir / f"{img_id}.jpg").write_bytes(r.content)
                f.write(json.dumps({
                    "id": img_id, "lat": p["lat"], "lon": p["lon"],
                    "heading": h, "pano_id": p["pano_id"],
                    "captured_at": p.get("date", ""),
                }) + "\n")
                n_new += 1
    print(f"[download] {n_new} new images -> {img_dir}")


def main():
    ap = argparse.ArgumentParser(description="Stage 3 — Street View grid")
    ap.add_argument("--bbox", action="store_true",
                    help="print the derived crawl bbox and exit")
    ap.add_argument("--scan", action="store_true", help="FREE metadata scan")
    ap.add_argument("--download", action="store_true",
                    help="Street View Static API download ($)")
    ap.add_argument("--max-panos", type=int, default=0,
                    help="cap panoramas for --download (cheap test runs)")
    args = ap.parse_args()

    if args.bbox or not (args.scan or args.download):
        b = crawl_bbox()
        print(f"crawl bbox (W,S,E,N) = "
              f"({b[0]:.5f}, {b[1]:.5f}, {b[2]:.5f}, {b[3]:.5f})")
        print(f"grid points @ {config.SV_GRID_M:.0f} m = "
              f"{len(grid_points(b))}")
    if args.scan:
        scan()
    if args.download:
        download(args.max_panos)


if __name__ == "__main__":
    main()
