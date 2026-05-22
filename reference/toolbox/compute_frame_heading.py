"""Compute per-frame heading from top-K visually-matched Mapillary compass.

For each frame in `frame_gps.jsonl`, look up the compass_angle of the matched
Mapillary frames (top_matches[].id), drop outliers (>90° from median), and
take the circular mean. Confidence buckets: <20° spread = high, 20-45° = med.

This is more reliable than estimating heading from consecutive-frame GPS
deltas — works on every frame standalone, doesn't require two consecutive
high-confidence GPS points.

Run once after frame_gps is built:
    python toolbox/compute_frame_heading.py
"""

import argparse
import json
import math
from pathlib import Path


def circ_mean(degs):
    rs = [math.radians(d) for d in degs]
    return math.degrees(math.atan2(sum(math.sin(r) for r in rs),
                                    sum(math.cos(r) for r in rs))) % 360


def circ_diff(a, b):
    return ((a - b + 540) % 360) - 180


def circ_std(degs):
    if not degs:
        return 360.0
    rs = [math.radians(d) for d in degs]
    sm = sum(math.sin(r) for r in rs) / len(rs)
    cm = sum(math.cos(r) for r in rs) / len(rs)
    R = math.sqrt(sm * sm + cm * cm)
    if R >= 0.999:
        return 0.0
    return math.degrees(math.sqrt(-2 * math.log(R)))


def filter_outliers(degs, max_diff=90.0):
    if len(degs) <= 2:
        return degs
    centre = circ_mean(degs)
    return [d for d in degs if abs(circ_diff(d, centre)) <= max_diff]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-gps", default="data/cities/zurich/frame_gps.jsonl")
    ap.add_argument("--mly-meta", default="data/cities/mapillary/zurich/meta.jsonl")
    ap.add_argument("--out", default="data/cities/zurich/frame_heading.jsonl")
    args = ap.parse_args()

    id2compass = {}
    with open(args.mly_meta) as f:
        for ln in f:
            r = json.loads(ln)
            if r.get("compass_angle") is not None:
                id2compass[r["id"]] = r["compass_angle"]
    print(f"[heading] {len(id2compass)} mly frames have compass_angle")

    n_in = n_out = 0
    by_conf = {"high": 0, "medium": 0, "low": 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.frame_gps) as fin, open(args.out, "w") as fout:
        for ln in fin:
            r = json.loads(ln)
            n_in += 1
            compasses = []
            for m in r.get("top_matches", []):
                c = id2compass.get(m["id"])
                if c is not None:
                    compasses.append(c)
            if len(compasses) < 3:
                continue
            kept = filter_outliers(compasses, max_diff=90.0)
            if len(kept) < 2:
                kept = compasses
            heading = round(circ_mean(kept), 1)
            spread = round(circ_std(kept), 1)
            if spread < 20:
                conf = "high"
            elif spread < 45:
                conf = "medium"
            else:
                conf = "low"
            by_conf[conf] += 1
            fout.write(json.dumps({
                "frame_id": r["frame_id"],
                "heading": heading,
                "heading_spread": spread,
                "heading_confidence": conf,
                "n_kept": len(kept),
                "n_total": len(compasses),
            }) + "\n")
            n_out += 1

    print(f"[heading] in={n_in}  out={n_out}  → {args.out}")
    print(f"  confidence: {by_conf}")


if __name__ == "__main__":
    main()
