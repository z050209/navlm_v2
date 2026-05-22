"""Step 10: cross-verify HMM-anchored GPS against VLM-visible POIs.

For each frame, decide:
    PASS_LANDMARK  — GPS-nearby POI ∩ VLM-seen POI ≠ ∅      (strong)
    PASS_STREET    — VLM saw nothing AND GPS not near POI   (weak: agreed empty)
    INCONCLUSIVE   — partial mismatch (one side empty / nearby mismatch)
    FAIL           — VLM saw a POI > VLM_MISMATCH_FAIL_M from GPS

Pure dict-and-set logic — no VLM call. Inputs are step 8 (HMM) and
step 9 (VLM POI scan).

Run:
    python -m pipeline.step_10_vlm_verify --video bahnhofstrasse
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import (  # noqa: E402
    VIDEO_NAMES, paths_for, POI_OSM,
    GPS_NEAR_RADIUS_M, GPS_VISIBLE_RADIUS_M, VLM_MISMATCH_FAIL_M,
)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(p1) * math.cos(p2) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


def load_poi_db():
    """Return {name: (lat, lon, radius_m)}.

    Combines OSM tier-1 landmarks with scenery POIs (each carries its
    own radius — streets ~300m, lake ~600m, etc.).
    """
    db = {}
    if POI_OSM.exists():
        with open(POI_OSM) as f:
            data = json.load(f)
        for name, p in data.items():
            db[name] = (p["lat"], p["lon"], GPS_VISIBLE_RADIUS_M)

    # scenery POIs override (their own radius)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "toolbox"))
    try:
        from scenery_pois import SCENERY_POIS  # noqa
        for name, info in SCENERY_POIS.items():
            db[name] = (info["lat"], info["lon"],
                        info.get("radius_m", GPS_VISIBLE_RADIUS_M))
    except ImportError:
        pass
    return db


def pois_within(lat, lon, db, radius):
    """Return list of POI names within `radius` of (lat, lon).

    For scenery POIs that have a custom radius, use the larger of
    `radius` and the POI's own radius (so big POIs like Lake Zurich
    aren't artificially restricted to 200m).
    """
    out = []
    for name, (plat, plon, prad) in db.items():
        d = haversine_m(lat, lon, plat, plon)
        effective = max(radius, prad)
        if d <= effective:
            out.append((name, d))
    out.sort(key=lambda x: x[1])
    return out


def verdict_for_frame(hmm_row, vlm_pois, db):
    """Compute verdict + evidence dict for one frame.

    hmm_row:    dict from step 8 (snap_lat, snap_lon, matched, ...)
    vlm_pois:   list of POI names Gemma saw in the frame (may be empty)
    db:         {name: (lat, lon, radius_m)}
    """
    if not hmm_row.get("matched"):
        return "FAIL", {"reason": "HMM did not match this frame"}

    lat, lon = hmm_row["snap_lat"], hmm_row["snap_lon"]
    near = [n for n, _ in pois_within(lat, lon, db, GPS_NEAR_RADIUS_M)]
    visible_range = [n for n, _ in pois_within(lat, lon, db, GPS_VISIBLE_RADIUS_M)]
    overlap = sorted(set(vlm_pois) & set(visible_range))

    if not vlm_pois:
        if not near:
            return "PASS_STREET", {
                "reason": "VLM saw no POIs; GPS is on a regular street segment",
                "gps_pois_within_50m":  near,
                "gps_pois_within_200m": visible_range,
            }
        return "INCONCLUSIVE", {
            "reason": "GPS at a POI but VLM didn't see it (camera angle)",
            "gps_pois_within_50m":  near,
            "gps_pois_within_200m": visible_range,
        }

    if overlap:
        return "PASS_LANDMARK", {
            "reason": f"VLM-seen {overlap} confirmed within GPS visible range",
            "overlap": overlap,
            "vlm_pois": vlm_pois,
            "gps_pois_within_200m": visible_range,
        }

    # VLM saw POIs, none in visible range. How far away are they?
    vlm_dists = []
    for v in vlm_pois:
        if v in db:
            vlm_dists.append((v, haversine_m(lat, lon, db[v][0], db[v][1])))
    if not vlm_dists:
        return "INCONCLUSIVE", {
            "reason": "VLM POIs not in DB",
            "vlm_pois": vlm_pois,
        }
    nearest = min(vlm_dists, key=lambda x: x[1])
    if nearest[1] > VLM_MISMATCH_FAIL_M:
        return "FAIL", {
            "reason": (f"VLM sees {nearest[0]} {nearest[1]:.0f}m from GPS — "
                       f"GPS likely wrong"),
            "nearest_vlm_poi": nearest[0],
            "nearest_vlm_poi_dist_m": round(nearest[1], 1),
            "vlm_pois": vlm_pois,
        }
    return "INCONCLUSIVE", {
        "reason": (f"VLM-GPS mismatch within {nearest[1]:.0f}m, ambiguous"),
        "nearest_vlm_poi": nearest[0],
        "nearest_vlm_poi_dist_m": round(nearest[1], 1),
        "vlm_pois": vlm_pois,
        "gps_pois_within_200m": visible_range,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, choices=VIDEO_NAMES)
    ap.add_argument("--variant", default="", help="output namespace suffix (e.g. _hq)")
    args = ap.parse_args()

    P = paths_for(args.video, variant=args.variant)
    if not P["step_08_hmm"].exists():
        sys.exit(f"missing step 8 output: {P['step_08_hmm']}")
    if not P["step_09_vlm_poi"].exists():
        sys.exit(f"missing step 9 output: {P['step_09_vlm_poi']}")

    # Load step 9 (VLM POI scan) keyed by frame_id
    vlm_by_fid = {}
    for ln in open(P["step_09_vlm_poi"]):
        r = json.loads(ln)
        vlm_by_fid[r["frame_id"]] = r.get("visible_pois", [])

    db = load_poi_db()
    print(f"[step10:{args.video}] poi_db={len(db)}  vlm_scanned_frames={len(vlm_by_fid)}")

    counts = {"PASS_LANDMARK": 0, "PASS_STREET": 0, "INCONCLUSIVE": 0, "FAIL": 0}
    n = 0
    with open(P["step_10_verify"], "w") as fout:
        for ln in open(P["step_08_hmm"]):
            hmm = json.loads(ln)
            fid = hmm["frame_id"]
            vlm_pois = vlm_by_fid.get(fid, [])  # may be []; that's fine
            verdict, evidence = verdict_for_frame(hmm, vlm_pois, db)
            counts[verdict] += 1
            n += 1
            fout.write(json.dumps({
                "frame_id": fid,
                "verdict":  verdict,
                "vlm_scanned": fid in vlm_by_fid,
                "evidence": evidence,
            }, ensure_ascii=False) + "\n")

    print(f"[step10:{args.video}] {n} frames, "
          + " ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
