"""Two-signal consensus filter on top-K visual matches.

The original visual_match_gps.py picks top-K Mapillary matches by visual
similarity and takes their median GPS. That can be wrong when the K matches
look similar visually (stone arches, generic façades) but are actually from
DIFFERENT physical locations across the city.

This script adds a *consensus* check using two signals from the matches:
  1. GPS dispersion among top-K — max pairwise haversine distance
  2. Compass spread among top-K — circular std of compass_angle

A frame is kept (high/medium) only if BOTH signals agree:
  - GPS dispersion small  → matches really are from one neighborhood
  - Compass spread small  → matches show similar viewpoint (not the same
                            building from opposite sides)

We also optionally re-aggregate using only the inlier subset (those within
75 m of the gps cluster median AND within 60° of the compass median).

Output frame_gps_refined.jsonl preserves all original fields, adds:
  - gps_dispersion_m       (max pairwise meters among top-K)
  - compass_spread_deg     (circular std of compass)
  - n_inliers              (count of matches that survive inlier filter)
  - confidence_v2          (high / medium / low — replaces 'confidence')
  - gps_v2                 (re-aggregated GPS from inliers, if any)
"""

import argparse
import json
import math
from pathlib import Path


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def gps_max_pairwise_m(pts):
    if len(pts) < 2:
        return 0.0
    return max(haversine_m(p1[0], p1[1], p2[0], p2[1])
               for i, p1 in enumerate(pts) for p2 in pts[i + 1:])


def circ_mean(degs):
    if not degs:
        return None
    rs = [math.radians(d) for d in degs]
    return math.degrees(math.atan2(sum(math.sin(r) for r in rs),
                                    sum(math.cos(r) for r in rs))) % 360


def circ_diff(a, b):
    return ((a - b + 540) % 360) - 180


def circ_std(degs):
    if len(degs) < 2:
        return 0.0
    rs = [math.radians(d) for d in degs]
    sm = sum(math.sin(r) for r in rs) / len(rs)
    cm = sum(math.cos(r) for r in rs) / len(rs)
    R = math.sqrt(sm * sm + cm * cm)
    if R >= 0.999:
        return 0.0
    return math.degrees(math.sqrt(-2 * math.log(R)))


def median_latlon(pts):
    pts = sorted(pts)
    return pts[len(pts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-gps", default="data/cities/zurich/frame_gps.jsonl")
    ap.add_argument("--mly-meta", default="data/cities/mapillary/zurich/meta.jsonl")
    ap.add_argument("--out", default="data/cities/zurich/frame_gps_refined.jsonl")
    ap.add_argument("--gps-disp-high-m", type=float, default=50.0)
    ap.add_argument("--gps-disp-med-m", type=float, default=120.0)
    ap.add_argument("--compass-spread-high-deg", type=float, default=30.0)
    ap.add_argument("--compass-spread-med-deg", type=float, default=60.0)
    ap.add_argument("--inlier-gps-m", type=float, default=75.0)
    ap.add_argument("--inlier-compass-deg", type=float, default=60.0)
    args = ap.parse_args()

    # mly id → compass
    id2compass = {}
    with open(args.mly_meta) as f:
        for ln in f:
            r = json.loads(ln)
            if r.get("compass_angle") is not None:
                id2compass[r["id"]] = r["compass_angle"]
    print(f"[refine] {len(id2compass)} mly compasses loaded")

    n_in = n_high = n_med = n_low = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.frame_gps) as fin, open(args.out, "w") as fout:
        for ln in fin:
            r = json.loads(ln)
            n_in += 1
            tops = r.get("top_matches", []) or []
            gps_pts = [(m["lat"], m["lon"]) for m in tops]
            compasses = [id2compass.get(m["id"]) for m in tops]
            compasses = [c for c in compasses if c is not None]

            gps_disp = gps_max_pairwise_m(gps_pts) if len(gps_pts) >= 2 else 0.0
            compass_spread = circ_std(compasses) if len(compasses) >= 2 else 999.0

            r["gps_dispersion_m"] = round(gps_disp, 1)
            r["compass_spread_deg"] = round(compass_spread, 1)

            # Inlier filter: within X m of gps median AND within Y° of compass median
            if gps_pts:
                gps_med = median_latlon(gps_pts)
                if compasses:
                    cmean = circ_mean(compasses)
                    inliers = [(m, c) for m, c in zip(tops, [id2compass.get(t["id"]) for t in tops])
                               if (haversine_m(m["lat"], m["lon"], *gps_med) <= args.inlier_gps_m
                                   and (c is not None and abs(circ_diff(c, cmean)) <= args.inlier_compass_deg))]
                else:
                    inliers = [(m, None) for m in tops
                               if haversine_m(m["lat"], m["lon"], *gps_med) <= args.inlier_gps_m]
                r["n_inliers"] = len(inliers)
                if inliers:
                    in_lats = [m["lat"] for m, _ in inliers]
                    in_lons = [m["lon"] for m, _ in inliers]
                    r["gps_v2"] = [round(sorted(in_lats)[len(in_lats)//2], 7),
                                   round(sorted(in_lons)[len(in_lons)//2], 7)]
                else:
                    r["gps_v2"] = None
            else:
                r["n_inliers"] = 0
                r["gps_v2"] = None

            # Confidence v2: BOTH signals must agree
            if (gps_disp <= args.gps_disp_high_m
                    and compass_spread <= args.compass_spread_high_deg
                    and r.get("n_inliers", 0) >= 3):
                r["confidence_v2"] = "high"
                n_high += 1
            elif (gps_disp <= args.gps_disp_med_m
                    and compass_spread <= args.compass_spread_med_deg
                    and r.get("n_inliers", 0) >= 3):
                r["confidence_v2"] = "medium"
                n_med += 1
            else:
                r["confidence_v2"] = "low"
                n_low += 1

            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[refine] in={n_in}")
    print(f"  v2 confidence: high={n_high}  medium={n_med}  low={n_low}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
