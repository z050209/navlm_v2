"""Download Mapillary images + GPS inside a city bounding box.

Prerequisites
-------------
1. Register at https://www.mapillary.com/developer and create a Client app.
2. Copy the client token (starts with 'MLY|...').
3. Export it: export MAPILLARY_TOKEN="MLY|xxx"

Usage
-----
    python toolbox/fetch_mapillary.py --city zurich \
        --bbox 8.528,47.366,8.557,47.385 \
        --limit 5000 \
        --size thumb_256_url

Output
------
    data/mapillary/<city>/meta.jsonl          one line per image:
        {"id": "...", "lat": 47.372, "lon": 8.541, "url": "...", "captured_at": "..."}
    data/mapillary/<city>/images/<id>.jpg     downloaded JPG thumbnails

Notes
-----
- Mapillary Graph API: https://www.mapillary.com/developer/api-documentation
- thumb_256_url is the smallest thumbnail (fast download, still enough for DINOv2)
- thumb_1024_url is higher-res but 16× the bandwidth
- Pagination is handled via the "next" cursor.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

ROOT = Path("/pub/evaluation_group/ning/test/navlm")
BASE = "https://graph.mapillary.com/images"


def fetch_page(token, bbox, size_field, fields, limit, after=None):
    """Mapillary Graph API requires OAuth header, not query-string token."""
    params = {
        "bbox": bbox,
        "fields": ",".join(fields),
        "limit": limit,
    }
    if after:
        params["after"] = after
    headers = {"Authorization": f"OAuth {token}"}
    r = requests.get(BASE, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--bbox", required=True,
                    help="minLon,minLat,maxLon,maxLat  (Zurich centre default: 8.528,47.366,8.557,47.385)")
    ap.add_argument("--limit", type=int, default=5000, help="total images cap")
    ap.add_argument("--page-size", type=int, default=500, help="per API call")
    ap.add_argument("--size", default="thumb_256_url",
                    choices=["thumb_256_url", "thumb_1024_url", "thumb_original_url"])
    ap.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN"),
                    help="or set MAPILLARY_TOKEN env var")
    args = ap.parse_args()

    if not args.token:
        raise SystemExit(
            "ERROR: no Mapillary token. Export MAPILLARY_TOKEN or pass --token. "
            "Register at https://www.mapillary.com/developer to get one."
        )

    out_dir = ROOT / "data" / "mapillary" / args.city
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"

    fields = ["id", "geometry", "captured_at", args.size]
    got = 0
    after = None
    with open(meta_path, "w") as meta_f:
        while got < args.limit:
            try:
                data = fetch_page(args.token, args.bbox, args.size, fields,
                                  min(args.page_size, args.limit - got), after)
            except requests.HTTPError as e:
                print(f"[mapillary] HTTP error {e.response.status_code}: {e.response.text[:300]}")
                break
            items = data.get("data", [])
            if not items:
                break
            for it in items:
                coord = it.get("geometry", {}).get("coordinates")
                url = it.get(args.size)
                if not coord or not url:
                    continue
                lon, lat = coord
                rec = {
                    "id": it["id"],
                    "lat": lat,
                    "lon": lon,
                    "captured_at": it.get("captured_at"),
                    "url": url,
                }
                # Download JPG
                try:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    (img_dir / f"{it['id']}.jpg").write_bytes(r.content)
                except Exception as e:
                    print(f"  skip {it['id']}: {e}")
                    continue
                meta_f.write(json.dumps(rec) + "\n")
                meta_f.flush()
                got += 1
                if got % 100 == 0:
                    print(f"  downloaded {got}...")
            # pagination
            paging = data.get("paging", {})
            cursor = paging.get("cursors", {}).get("after")
            if not cursor:
                break
            after = cursor
            time.sleep(0.1)  # be polite
    print(f"[mapillary] {got} images + GPS saved under {out_dir}")


if __name__ == "__main__":
    main()
