#!/usr/bin/env python3
"""Download Google Street View images around the tourist POIs found by
the VLM video scan.

For each POI we know a hand-verified (lat, lon). We sample the POI centre
plus a ring of nearby points, ask the FREE Street View metadata endpoint
whether a panorama exists there, and if so download a Street View Static
image with the camera heading aimed at the POI.

Run from navlm_ss/:
    python fetch_streetview_poi.py

Output: ../preview/streetview_poi/<POI>/*.jpg  +  manifest.csv
"""

import csv
import json
import math
import os
import time
from pathlib import Path

import requests

from toolbox.zurich_landmarks_gps import ZURICH_LANDMARKS
from toolbox.scenery_pois import SCENERY_POIS

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")  # set via .env / env var

SV_META = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_IMG = "https://maps.googleapis.com/maps/api/streetview"
OSM_TABLE = Path("data/cities/zurich/landmarks_zurich_osm.json")
VLM_SCAN = Path("data/cities/zurich/_video_poi_multi.jsonl")
OUT = Path("../preview/streetview_poi")

IMG_SIZE = "640x640"
FOV = 90
RING_M = 35          # radius of the sampled ring around each POI
RING_BEARINGS = [0, 90, 180, 270]   # N, E, S, W sample points


def haversine_m(la1, lo1, la2, lo2):
    R = 6371000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlam = math.radians(lo2 - lo1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(la1, lo1, la2, lo2):
    """Initial compass bearing from point 1 to point 2."""
    p1, p2 = math.radians(la1), math.radians(la2)
    dl = math.radians(lo2 - lo1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def offset_latlon(lat, lon, bearing, dist_m):
    """Point dist_m metres from (lat,lon) along compass bearing."""
    R = 6371000.0
    br = math.radians(bearing)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    dr = dist_m / R
    p2 = math.asin(math.sin(p1) * math.cos(dr) +
                   math.cos(p1) * math.sin(dr) * math.cos(br))
    l2 = l1 + math.atan2(math.sin(br) * math.sin(dr) * math.cos(p1),
                         math.cos(dr) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


def resolve_pois():
    """Map the 25 VLM-scanned POI names -> (lat, lon, source)."""
    osm = json.loads(OSM_TABLE.read_text(encoding="utf-8"))
    scan_pois = set()
    for ln in VLM_SCAN.open(encoding="utf-8"):
        scan_pois.update(json.loads(ln).get("visible_pois", []))

    resolved = {}
    for name in sorted(scan_pois):
        if name in ZURICH_LANDMARKS:
            lat, lon, _ = ZURICH_LANDMARKS[name]
            resolved[name] = (lat, lon, "curated")
        elif name in SCENERY_POIS:
            s = SCENERY_POIS[name]
            resolved[name] = (s["lat"], s["lon"], "scenery")
        elif name in osm:
            resolved[name] = (osm[name]["lat"], osm[name]["lon"], "osm")
        else:
            # loose match against OSM keys
            hit = next((k for k in osm if name.lower() in k.lower()), None)
            if hit:
                resolved[name] = (osm[hit]["lat"], osm[hit]["lon"], "osm~")
            else:
                resolved[name] = (None, None, "UNRESOLVED")
    return resolved


def sv_metadata(lat, lon):
    r = requests.get(SV_META, params={
        "location": f"{lat},{lon}", "key": API_KEY, "source": "outdoor",
    }, timeout=30)
    return r.json()


def download_sv(pano_lat, pano_lon, heading, dest):
    r = requests.get(SV_IMG, params={
        "size": IMG_SIZE, "location": f"{pano_lat},{pano_lon}",
        "heading": round(heading, 1), "fov": FOV, "pitch": 0,
        "source": "outdoor", "key": API_KEY,
    }, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return len(r.content)


def main():
    pois = resolve_pois()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    print(f"Resolved {len(pois)} POIs. Fetching Street View...\n")

    n_img = 0
    for name, (lat, lon, src) in pois.items():
        if lat is None:
            print(f"-- {name}: UNRESOLVED, skipped")
            continue
        safe = name.replace(" ", "_").replace("/", "_")
        poi_dir = OUT / safe
        poi_dir.mkdir(exist_ok=True)

        # sample points: POI centre + a ring around it
        samples = [("center", lat, lon)]
        for b in RING_BEARINGS:
            slat, slon = offset_latlon(lat, lon, b, RING_M)
            samples.append((f"ring{b:03d}", slat, slon))

        got = 0
        for tag, slat, slon in samples:
            meta = sv_metadata(slat, slon)
            status = meta.get("status")
            if status != "OK":
                continue
            ploc = meta.get("location", {})
            plat, plon = ploc.get("lat"), ploc.get("lng")
            date = meta.get("date", "")
            pano = meta.get("pano_id", "")
            # aim camera from the panorama back at the POI centre
            head = bearing_deg(plat, plon, lat, lon)
            dist = haversine_m(plat, plon, lat, lon)
            dest = poi_dir / f"{safe}_{tag}.jpg"
            try:
                kb = download_sv(plat, plon, head, dest) // 1024
            except Exception as e:
                print(f"   {name}/{tag}: dl failed {e}")
                continue
            got += 1
            n_img += 1
            manifest.append({
                "poi": name, "gps_source": src, "sample": tag,
                "poi_lat": lat, "poi_lon": lon,
                "pano_lat": plat, "pano_lon": plon, "pano_id": pano,
                "capture_date": date, "heading_to_poi": round(head, 1),
                "dist_m": round(dist, 1), "file": str(dest),
            })
            time.sleep(0.05)
        print(f"{name:20s} [{src:8s}]  {got}/{len(samples)} panos")

    mf = OUT / "manifest.csv"
    with mf.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"\n=== done: {n_img} images, manifest -> {mf} ===")


if __name__ == "__main__":
    main()
