#!/usr/bin/env python3
"""Dense Google Street View crawl over central Zurich — builds a
clean, pedestrian-level replacement for the noisy Mapillary GPS index.

Strategy
--------
1. Lay a grid over the old-town bbox (point every GRID_M metres).
2. At each grid point call the FREE Street View *metadata* endpoint;
   it returns the nearest panorama's id + exact GPS + capture date.
3. Deduplicate by pano_id (neighbouring grid points hit the same pano).
4. For each unique pano, download HEADINGS flat crops via the Static API.
5. Write meta.jsonl in a Mapillary-compatible schema (one row per image).

Cost: metadata is FREE; Static API is $7 / 1000 images.
Fully resumable — re-running skips pano_ids already in meta.jsonl.

Run from navlm_ss/:
    python fetch_streetview_grid.py --grid 50 --headings 0,90,180,270 --max-panos 800
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")  # set via .env / env var

SV_META = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMG = "https://maps.googleapis.com/maps/api/streetview"

# old-town bbox (W, S, E, N) — same box used by the Mapillary fetcher
BBOX = (8.523, 47.358, 8.557, 47.382)
OUT = Path("data/cities/streetview/zurich")
IMG_SIZE = "640x640"
FOV = 90
PITCH = 0
LAT_M = 111320.0


def grid_points(bbox, grid_m):
    w, s, e, n = bbox
    lon_m = LAT_M * math.cos(math.radians((s + n) / 2))
    nx = max(1, int((e - w) * lon_m / grid_m))
    ny = max(1, int((n - s) * LAT_M / grid_m))
    pts = []
    for ix in range(nx + 1):
        for iy in range(ny + 1):
            pts.append((s + (n - s) * iy / ny, w + (e - w) * ix / nx))
    return pts, nx, ny


def sv_metadata(lat, lon, session):
    r = session.get(SV_META, params={
        "location": f"{lat},{lon}", "key": API_KEY, "source": "outdoor",
    }, timeout=30)
    return r.json()


def download_sv(pano_id, heading, dest, session):
    r = session.get(SV_IMG, params={
        "size": IMG_SIZE, "pano": pano_id, "heading": heading,
        "fov": FOV, "pitch": PITCH, "key": API_KEY,
    }, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return len(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=50.0,
                    help="grid spacing in metres")
    ap.add_argument("--headings", default="0,90,180,270",
                    help="comma-separated camera headings per pano")
    ap.add_argument("--max-panos", type=int, default=0,
                    help="cap on unique panos to download (0 = no cap)")
    ap.add_argument("--sub-bbox", default="",
                    help="restrict downloads to W,S,E,N (panos outside skipped)")
    ap.add_argument("--scan-only", action="store_true",
                    help="only do the FREE metadata scan, no image downloads")
    ap.add_argument("--skip-scan", action="store_true",
                    help="reuse cached panos.jsonl, skip the metadata scan")
    args = ap.parse_args()
    headings = [int(h) for h in args.headings.split(",")]
    sub_bbox = [float(x) for x in args.sub_bbox.split(",")] if args.sub_bbox else None

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "meta.jsonl"
    panos_path = OUT / "panos.jsonl"   # one row per unique pano (scan cache)

    # resume: panos already scanned, images already in meta
    scanned = {}
    if panos_path.exists():
        for ln in panos_path.open(encoding="utf-8"):
            d = json.loads(ln)
            scanned[d["pano_id"]] = d
    done_imgs = set()
    if meta_path.exists():
        for ln in meta_path.open(encoding="utf-8"):
            done_imgs.add(json.loads(ln)["id"])

    pts, nx, ny = grid_points(BBOX, args.grid)
    n_static = len(headings) * (args.max_panos or 9999)
    print(f"=== Street View grid crawl ===")
    print(f"  bbox        : {BBOX}")
    print(f"  grid        : {args.grid} m  ->  {nx+1}x{ny+1} = {len(pts)} points")
    print(f"  headings    : {headings}  ({len(headings)} imgs/pano)")
    print(f"  max panos   : {args.max_panos or 'unlimited'}")
    print(f"  resume      : {len(scanned)} panos scanned, {len(done_imgs)} imgs done")
    print(f"  scan-only   : {args.scan_only}")
    print()

    session = requests.Session()
    fpanos = panos_path.open("a", encoding="utf-8")
    t0 = time.time()

    # ---- Phase 1: metadata scan (FREE) ----
    for i, (lat, lon) in enumerate([] if args.skip_scan else pts, 1):
        if i % 200 == 0:
            print(f"  scan [{i}/{len(pts)}]  unique panos={len(scanned)}")
        m = sv_metadata(lat, lon, session)
        if m.get("status") != "OK":
            continue
        pid = m.get("pano_id")
        if not pid or pid in scanned:
            continue
        loc = m.get("location", {})
        rec = {"pano_id": pid, "lat": loc.get("lat"), "lon": loc.get("lng"),
               "date": m.get("date", ""), "copyright": m.get("copyright", "")}
        scanned[pid] = rec
        fpanos.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fpanos.flush()
    fpanos.close()
    print(f"\n  scan done: {len(scanned)} unique panos "
          f"({time.time()-t0:.0f}s, all FREE)\n")

    if args.scan_only:
        print("  --scan-only set; stopping before image downloads.")
        return

    # ---- Phase 2: download Static API crops ($7/1000) ----
    panos = list(scanned.values())
    if sub_bbox:
        w, s, e, n = sub_bbox
        panos = [p for p in panos if w <= p["lon"] <= e and s <= p["lat"] <= n]
        print(f"  sub-bbox filter {sub_bbox}: {len(panos)} panos in box")
    if args.max_panos:
        panos = panos[:args.max_panos]
    n_target = len(panos) * len(headings)
    print(f"  downloading up to {n_target} images "
          f"(~${n_target*7/1000:.2f} on Static API)\n")

    fmeta = meta_path.open("a", encoding="utf-8")
    n_dl = n_skip = 0
    for j, p in enumerate(panos, 1):
        for h in headings:
            img_id = f"{p['pano_id']}_h{h:03d}"
            if img_id in done_imgs:
                n_skip += 1
                continue
            dest = img_dir / f"{img_id}.jpg"
            try:
                download_sv(p["pano_id"], h, dest, session)
            except Exception as e:
                print(f"   {img_id}: dl failed {e}")
                continue
            # Mapillary-compatible row
            fmeta.write(json.dumps({
                "id": img_id, "lat": p["lat"], "lon": p["lon"],
                "compass_angle": h, "captured_at": p["date"],
                "is_pano": False, "pano_id": p["pano_id"],
                "fov": FOV, "pitch": PITCH, "source": "google_streetview",
            }, ensure_ascii=False) + "\n")
            fmeta.flush()
            n_dl += 1
        if j % 100 == 0:
            print(f"  dl [{j}/{len(panos)} panos]  new={n_dl} skip={n_skip}")
    fmeta.close()

    print(f"\n=== done ===")
    print(f"  unique panos     : {len(scanned)}")
    print(f"  images downloaded: {n_dl}  (skipped {n_skip} already done)")
    print(f"  est. Static cost : ${n_dl*7/1000:.2f}")
    print(f"  images -> {img_dir}")
    print(f"  meta   -> {meta_path}  (Mapillary-compatible schema)")


if __name__ == "__main__":
    main()
