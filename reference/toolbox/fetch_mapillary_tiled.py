"""Tile-based Mapillary crawler.

Mapillary's Graph API caps a single bbox query at ~5000 results, so a single
big-bbox query gives you a SPARSE sample of the city, not dense coverage.
This script splits the bbox into small tiles and queries each separately,
then deduplicates by frame ID.

Run on a host that can reach graph.mapillary.com (your laptop, not the
firewalled server).

Usage
-----
    export MAPILLARY_TOKEN="MLY|xxx"
    python toolbox/fetch_mapillary_tiled.py \\
        --city zurich_tiled \\
        --bbox 8.480,47.340,8.600,47.420 \\
        --tile-size-m 400 \\
        --per-tile-limit 1000

Estimated cost (Zurich central area 8km × 6km):
    400m tiles  →  20 × 15 = 300 tiles
    1000 frames/tile cap  →  up to 300k unique frames
    download time  →  ~2-4 hours on a home connection

After download, rsync to server.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

LAT_M = 111000.0  # 1° lat
LON_M = 78000.0   # 1° lon at ~47°N


def graph_get(token, endpoint, params, retries=3):
    """GET with simple retry."""
    headers = {"Authorization": f"OAuth {token}"}
    url = f"https://graph.mapillary.com/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                wait = 2 ** attempt * 5
                print(f"  rate-limited, sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"  error: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            time.sleep(5)
    return None


def fetch_tile(token, w, s, e, n, limit):
    """Fetch metadata for one tile. Returns list of dicts."""
    params = {
        "bbox": f"{w},{s},{e},{n}",
        "fields": "id,geometry,captured_at,compass_angle,sequence,is_pano,thumb_256_url,thumb_1024_url",
        "limit": min(limit, 2000),
    }
    out = []
    cursor = None
    pages = 0
    while True:
        if cursor:
            params["after"] = cursor
        data = graph_get(token, "images", params)
        if not data or "data" not in data:
            break
        for item in data["data"]:
            geom = item.get("geometry", {}).get("coordinates")
            if not geom or len(geom) < 2:
                continue
            out.append({
                "id": item["id"],
                "lon": geom[0],
                "lat": geom[1],
                "captured_at": item.get("captured_at"),
                "compass_angle": item.get("compass_angle"),
                "is_pano": item.get("is_pano", False),
                "sequence": item.get("sequence"),
                "thumb_256_url": item.get("thumb_256_url"),
                "thumb_1024_url": item.get("thumb_1024_url"),
            })
            if len(out) >= limit:
                return out
        pages += 1
        cursor = data.get("paging", {}).get("cursors", {}).get("after")
        if not cursor or pages >= 10:
            break
    return out


def download_image(url, dest_path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    try:
        r = requests.get(url, timeout=30, stream=True)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"  dl-err: {type(e).__name__}", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--bbox", required=True, help="W,S,E,N")
    ap.add_argument("--tile-size-m", type=float, default=400.0)
    ap.add_argument("--per-tile-limit", type=int, default=1000)
    ap.add_argument("--out-dir", default="data/mapillary")
    ap.add_argument("--size", choices=["256", "1024"], default="256")
    ap.add_argument("--skip-download", action="store_true",
                    help="only fetch metadata, no image download")
    args = ap.parse_args()

    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        print("ERROR: set MAPILLARY_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    w, s, e, n = map(float, args.bbox.split(","))
    # Tile bbox in approximate meters
    n_lon = max(1, int(round((e - w) * LON_M / args.tile_size_m)))
    n_lat = max(1, int(round((n - s) * LAT_M / args.tile_size_m)))
    print(f"[crawl] tiling {n_lon} × {n_lat} = {n_lon * n_lat} tiles")

    out_dir = Path(args.out_dir) / args.city
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"

    # Resume: skip ids already in meta.jsonl
    seen_ids = set()
    if meta_path.exists():
        for ln in open(meta_path):
            try:
                seen_ids.add(json.loads(ln)["id"])
            except Exception:
                pass
        print(f"[crawl] resume: {len(seen_ids)} ids already in meta.jsonl")

    n_meta_new = n_dl_new = 0
    t0 = time.time()
    url_field = f"thumb_{args.size}_url"
    fmeta = open(meta_path, "a")

    for ix in range(n_lon):
        for iy in range(n_lat):
            tw = w + (e - w) * ix / n_lon
            te = w + (e - w) * (ix + 1) / n_lon
            ts = s + (n - s) * iy / n_lat
            tn = s + (n - s) * (iy + 1) / n_lat

            print(f"[tile {ix},{iy}] bbox=({tw:.4f},{ts:.4f},{te:.4f},{tn:.4f})")
            items = fetch_tile(token, tw, ts, te, tn, args.per_tile_limit)
            print(f"  → {len(items)} items")

            for item in items:
                if item["id"] in seen_ids:
                    continue
                seen_ids.add(item["id"])
                fmeta.write(json.dumps(item, ensure_ascii=False) + "\n")
                fmeta.flush()
                n_meta_new += 1
                if args.skip_download:
                    continue
                url = item.get(url_field)
                if url:
                    img_path = img_dir / f"{item['id']}.jpg"
                    if download_image(url, img_path):
                        n_dl_new += 1

            elapsed = time.time() - t0
            done = ix * n_lat + iy + 1
            total = n_lon * n_lat
            print(f"  [{done}/{total}] new_meta={n_meta_new} new_dl={n_dl_new} "
                  f"unique_total={len(seen_ids)} elapsed={elapsed:.0f}s")

    fmeta.close()
    print(f"[crawl] done. {len(seen_ids)} unique ids, "
          f"{n_meta_new} new meta, {n_dl_new} new images")
    print(f"  meta:   {meta_path}")
    print(f"  images: {img_dir}")
    if args.skip_download:
        print("  (--skip-download: re-run without it to fetch images)")


if __name__ == "__main__":
    main()
