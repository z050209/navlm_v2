#!/usr/bin/env python3
"""Standalone Mapillary fetcher — copy to your laptop and run there.

Mapillary's API caps a single bbox query at ~5000 results. This script
splits Zurich's old town into small tiles (200m × 200m) and queries each
separately, then deduplicates by frame ID.

Single-file, no imports from the rest of the repo. Only needs `requests`.
Resumable: re-running picks up where it left off.

USAGE
-----
    pip install requests
    python fetch_zurich_dense.py

After it finishes, rsync the output to the server:
    rsync -avz data/mapillary/zurich_dense10k/ \\
        user@server:/pub/evaluation_group/ning/test/navlm/data/mapillary/zurich_dense10k/

CONFIG
------
Edit BBOX / TILE_SIZE_M / PER_TILE_LIMIT below if you want different
coverage. Defaults: 25 km² of central Zurich, ~625 tiles, up to ~30k images.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# ─────────────────────── CONFIG ───────────────────────

# Mapillary token — set the MAPILLARY_TOKEN env var (see .env.example).
MAPILLARY_TOKEN = os.environ.get("MAPILLARY_TOKEN", "")

# Output directory (relative to where you run the script)
OUT_DIR = Path("data/mapillary/zurich_dense10k")

# Bounding box: West, South, East, North (decimal degrees)
# 2.5 km × 2.5 km, shifted ~300 m east to capture more of Niederdorf /
# Limmat / Bellevue side of the old town.
BBOX = (8.528, 47.364, 8.560, 47.386)

# Each query tile size in metres. Smaller = denser sampling but more API calls.
TILE_SIZE_M = 200

# Max images per tile (Mapillary caps at ~2000 anyway)
PER_TILE_LIMIT = 1500

# Image variant: "256" = thumb_256_url (~30 KB each, fast, enough for DINOv2)
#                "1024" = thumb_1024_url (~250 KB each)
SIZE = "256"

# Skip downloading images, only fetch metadata.jsonl. Useful for first pass.
SKIP_DOWNLOAD = False

# ────────────────────────────────────────────────────────


LAT_M = 111000.0  # 1 degree latitude in metres
LON_M = 78000.0   # 1 degree longitude at ~47°N


def graph_get(endpoint, params, retries=3):
    """GET https://graph.mapillary.com/<endpoint> with retry."""
    headers = {"Authorization": f"OAuth {MAPILLARY_TOKEN}"}
    url = f"https://graph.mapillary.com/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                wait = 2 ** attempt * 5
                print(f"  rate-limited ({r.status_code}); sleeping {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"  err: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            time.sleep(5)
    return None


def fetch_tile(w, s, e, n, limit):
    """Pull one bbox of metadata. Paginates via 'after' cursor."""
    params = {
        "bbox": f"{w},{s},{e},{n}",
        "fields": "id,geometry,captured_at,compass_angle,sequence,is_pano,thumb_256_url,thumb_1024_url",
        "limit": min(limit, 2000),
    }
    out, cursor, pages = [], None, 0
    while True:
        if cursor:
            params["after"] = cursor
        data = graph_get("images", params)
        if not data or "data" not in data:
            break
        for item in data["data"]:
            geom = item.get("geometry", {}).get("coordinates")
            if not geom or len(geom) < 2:
                continue
            out.append({
                "id": item["id"],
                "lon": geom[0], "lat": geom[1],
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


def download_image(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = requests.get(url, timeout=30, stream=True)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"  dl-err: {type(e).__name__}", file=sys.stderr)
    return False


def main():
    assert MAPILLARY_TOKEN.startswith("MLY|"), "set MAPILLARY_TOKEN"
    w, s, e, n = BBOX
    n_lon = max(1, int(round((e - w) * LON_M / TILE_SIZE_M)))
    n_lat = max(1, int(round((n - s) * LAT_M / TILE_SIZE_M)))
    total_tiles = n_lon * n_lat
    print(f"\n=== Mapillary tile crawl ===")
    print(f"  bbox        : W={w} S={s} E={e} N={n}")
    print(f"  area        : {(e-w)*LON_M/1000:.1f} × {(n-s)*LAT_M/1000:.1f} km")
    print(f"  tile size   : {TILE_SIZE_M} m")
    print(f"  tile grid   : {n_lon} × {n_lat} = {total_tiles} tiles")
    print(f"  per-tile cap: {PER_TILE_LIMIT}")
    print(f"  output      : {OUT_DIR.resolve()}")
    print()

    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "meta.jsonl"

    # Resume: track ids already in meta.jsonl
    seen = set()
    if meta_path.exists():
        for ln in open(meta_path):
            try:
                seen.add(json.loads(ln)["id"])
            except Exception:
                pass
        print(f"  resuming   : {len(seen)} ids already in meta.jsonl")

    n_meta_new = n_dl_new = 0
    t0 = time.time()
    fmeta = open(meta_path, "a")

    for ix in range(n_lon):
        for iy in range(n_lat):
            tw = w + (e - w) * ix / n_lon
            te = w + (e - w) * (ix + 1) / n_lon
            ts = s + (n - s) * iy / n_lat
            tn = s + (n - s) * (iy + 1) / n_lat

            done = ix * n_lat + iy + 1
            print(f"\n[tile {ix:2d},{iy:2d}]  bbox=({tw:.4f},{ts:.4f},{te:.4f},{tn:.4f})")
            items = fetch_tile(tw, ts, te, tn, PER_TILE_LIMIT)
            print(f"  → {len(items)} items returned")

            for item in items:
                if item["id"] in seen:
                    continue
                seen.add(item["id"])
                fmeta.write(json.dumps(item, ensure_ascii=False) + "\n")
                fmeta.flush()
                n_meta_new += 1
                if SKIP_DOWNLOAD:
                    continue
                url_field = f"thumb_{SIZE}_url"
                url = item.get(url_field)
                if url:
                    img = img_dir / f"{item['id']}.jpg"
                    if download_image(url, img):
                        n_dl_new += 1

            elapsed = time.time() - t0
            rate = (n_meta_new / elapsed) if elapsed > 0 else 0
            print(f"  [{done}/{total_tiles}] new_meta={n_meta_new} "
                  f"new_dl={n_dl_new} unique={len(seen)} "
                  f"elapsed={elapsed:.0f}s rate={rate:.1f}/s")

    fmeta.close()
    print(f"\n=== DONE ===")
    print(f"  unique ids       : {len(seen)}")
    print(f"  new metadata rows: {n_meta_new}")
    print(f"  new images dl'd  : {n_dl_new}")
    print(f"  meta             : {meta_path}")
    print(f"  images           : {img_dir}")
    print(f"\nNext step (rsync to server):")
    print(f"  rsync -avz {OUT_DIR}/  user@server:/pub/evaluation_group/ning/test/navlm/{OUT_DIR}/")


if __name__ == "__main__":
    main()
