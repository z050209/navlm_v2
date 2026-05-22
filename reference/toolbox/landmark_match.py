"""Match OCR text (from ocr_augment.py) to Zurich landmarks with known GPS.

Produces a frame_gps-style JSONL of frames for which we have a
*high-precision* location fix (because a named sign was readable).

Usage
-----
    python toolbox/landmark_match.py \\
        --ocr data/cities/zurich/frame_ocr.jsonl \\
        --out data/cities/zurich/frame_gps_ocr.jsonl

Output row:
    {"frame_id": "frame_00123", "gps": [lat, lon], "confidence": "high",
     "source": "ocr", "matched": ["Grossmünster"], "texts_seen": [...]}
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zurich_landmarks_gps import ZURICH_LANDMARKS, build_alias_index  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    idx = build_alias_index()
    print(f"[match] {len(ZURICH_LANDMARKS)} landmarks, {len(idx)} aliases loaded")

    n_in = n_matched = 0
    with open(args.ocr) as fin, open(args.out, "w") as fout:
        for ln in fin:
            if not ln.strip():
                continue
            r = json.loads(ln)
            n_in += 1
            texts = [t.lower() for t in r.get("texts", [])]
            joined = " ".join(texts)
            matched_landmarks = []
            # Prefer specific (longer) aliases over generic ones
            for alias in sorted(idx.keys(), key=lambda a: -len(a)):
                if alias in joined:
                    canonical, lat, lon = idx[alias]
                    if canonical not in [m[0] for m in matched_landmarks]:
                        matched_landmarks.append((canonical, lat, lon))
            if not matched_landmarks:
                continue
            # If multiple landmarks matched, use the average of top-3
            top = matched_landmarks[:3]
            lat = sum(m[1] for m in top) / len(top)
            lon = sum(m[2] for m in top) / len(top)
            fout.write(json.dumps({
                "frame_id": r["frame_id"],
                "gps": [lat, lon],
                "confidence": "high",
                "source": "ocr",
                "matched": [m[0] for m in top],
                "texts_seen": r.get("texts", []),
            }, ensure_ascii=False) + "\n")
            n_matched += 1

    print(f"[match] {n_matched} / {n_in} frames got a landmark match")
    print(f"[match] wrote → {args.out}")


if __name__ == "__main__":
    main()
