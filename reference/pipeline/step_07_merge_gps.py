"""Step 7: merge per-frame GPS from visual matching and OCR.

Input  (per video):
    visual_refined  — output of refine_visual_match.py
                      (top-5 Mapillary matches with consensus filtering)
    ocr_match       — output of landmark_match.py (PaddleOCR + landmark table)

Output:
    one row per frame, with the best available GPS:
    {
      "frame_id": ...,
      "gps":         [lat, lon],
      "gps_source":  "ocr" | "visual_high" | "visual_mid" | "visual_low",
      "evidence":    {...source-specific fields...}
    }

Priority (highest wins):
    1. ocr (matched a landmark name)              → very reliable text anchor
    2. visual_refined.confidence_v2 = high        → DINOv2 top-5 agree
    3. visual_refined.confidence_v2 = mid
    4. visual_refined.confidence_v2 = low

Run:
    python -m pipeline.step_07_merge_gps --video bahnhofstrasse
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import VIDEO_NAMES, paths_for  # noqa: E402


def load_jsonl(p):
    if not Path(p).exists():
        return {}
    out = {}
    for ln in open(p):
        try:
            r = json.loads(ln)
            out[r["frame_id"]] = r
        except Exception:
            pass
    return out


import math

# Old-town centroid; reject any GPS more than BBOX_KM away as an outlier.
# This catches OCR landmark mismatches like "Zurich Airport" being read
# off a hotel-shuttle ad in a downtown frame.
OLD_TOWN_CENTER = (47.374, 8.541)
BBOX_KM = 1.5


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _within_bbox(gps):
    return _haversine_km(gps[0], gps[1], *OLD_TOWN_CENTER) <= BBOX_KM


def merge_one(visual_refined, ocr_match, drop_visual_low=True,
              apply_bbox=True):
    """Yield merged rows. visual_refined is the canonical frame set.

    Policy (strictest setting — only the cleanest frames survive):
      - GPS comes ONLY from `confidence_v2 = high` visual matches:
          top-5 Mapillary neighbours within 50m of each other AND
          their compass angles within 30° AND n_inliers ≥ 3.
      - visual_medium and visual_low are dropped.
      - OCR-alone frames are dropped — landmark name in a sign doesn't
        prove the camera is at that landmark.
      - When OCR also matches, the matched landmark is recorded in
        evidence as auxiliary signal but does NOT supply GPS.
      - apply_bbox=True: any GPS outside 1.5km of old-town center is
        dropped (sanity check, prevents OCR / visual outliers).
        Set to False for HQ runs where the index is already constrained.
    """
    for fid, vr in sorted(visual_refined.items(),
                          key=lambda kv: kv[0]):
        gps = vr.get("gps_v2") or vr.get("gps")
        if not gps:
            continue
        conf = vr.get("confidence_v2") or vr.get("confidence", "low")
        # Strict: only visual_high survives
        if conf != "high":
            continue

        # bbox sanity check (optional)
        if apply_bbox and not _within_bbox(gps):
            continue

        ocr = ocr_match.get(fid)
        evidence = {
            "n_inliers": vr.get("n_inliers"),
            "gps_dispersion_m": vr.get("gps_dispersion_m"),
            "compass_spread_deg": vr.get("compass_spread_deg"),
            "top1_sim": vr["top_matches"][0]["sim"]
                        if vr.get("top_matches") else None,
        }
        if ocr and ocr.get("matched"):
            evidence["ocr_matched"] = ocr["matched"]
            evidence["ocr_texts"] = ocr.get("texts_seen", [])

        yield {
            "frame_id": fid,
            "gps": gps,
            "gps_source": f"visual_{conf}",
            "evidence": evidence,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, choices=VIDEO_NAMES)
    ap.add_argument("--variant", default="", help="output namespace suffix (e.g. _hq)")
    ap.add_argument("--no-bbox", action="store_true",
                    help="disable the 1.5km old-town bbox sanity filter "
                         "(use when the Mapillary index is already constrained)")
    args = ap.parse_args()

    P = paths_for(args.video, variant=args.variant)
    P["out_dir"].mkdir(parents=True, exist_ok=True)

    visual = load_jsonl(P["legacy_visual_refined"])
    ocr    = load_jsonl(P["legacy_ocr_match"])
    print(f"[step7:{args.video}] visual={len(visual)} ocr={len(ocr)}  "
          f"bbox_filter={'ON' if not args.no_bbox else 'OFF'}")

    n_total = n_ocr = 0
    src_counts = {}
    with open(P["step_07_merged"], "w") as fout:
        for row in merge_one(visual, ocr, apply_bbox=not args.no_bbox):
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_total += 1
            if row["gps_source"] == "ocr":
                n_ocr += 1
            src_counts[row["gps_source"]] = src_counts.get(row["gps_source"], 0) + 1

    print(f"[step7:{args.video}] wrote {n_total} rows → {P['step_07_merged']}")
    print(f"  by source: {dict(sorted(src_counts.items()))}")


if __name__ == "__main__":
    main()
