"""Step 11: build per-video trusted_starts using GPS source × verdict matrix.

Joins step 7 (merged GPS + source) with step 8 (HMM snap) and step 10
(verdict). For each frame, looks up TRUSTED_MATRIX[gps_source][verdict]
to decide whether to keep it.

Output schema matches the existing `frame_starts_trusted.jsonl` format
so downstream consumers (synth_unified, viewer) work unchanged:

    {
      "frame_id":             ...,
      "image":                "<absolute path>",
      "gps":                  [lat, lon],
      "gps_source":           "ocr" | "visual_high" | ...,
      "heading":              <deg or null>,
      "heading_confidence":   "high" | "mid" | "low",
      "evidence":             {...source evidence + verdict...},
    }

Run:
    python -m pipeline.step_11_trusted --video bahnhofstrasse
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import VIDEO_NAMES, paths_for, TRUSTED_MATRIX  # noqa: E402


def load_jsonl_by_id(p):
    out = {}
    if not Path(p).exists():
        return out
    for ln in open(p):
        try:
            r = json.loads(ln)
            out[r["frame_id"]] = r
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, choices=VIDEO_NAMES)
    ap.add_argument("--variant", default="", help="output namespace suffix (e.g. _hq)")
    args = ap.parse_args()

    P = paths_for(args.video, variant=args.variant)
    for k in ("step_07_merged", "step_08_hmm", "step_10_verify"):
        if not P[k].exists():
            sys.exit(f"missing {k}: {P[k]}")

    merged   = load_jsonl_by_id(P["step_07_merged"])
    hmm      = load_jsonl_by_id(P["step_08_hmm"])
    verified = load_jsonl_by_id(P["step_10_verify"])
    heading  = load_jsonl_by_id(P["legacy_heading"])

    counts = {"kept": 0, "dropped_no_verdict": 0, "dropped_matrix": 0,
              "dropped_heading": 0,
              "by_source": {}, "by_verdict": {}}
    n_total = 0
    with open(P["step_11_trusted"], "w") as fout:
        for fid, m in merged.items():
            n_total += 1
            v = verified.get(fid)
            if not v:
                counts["dropped_no_verdict"] += 1
                continue
            verdict = v["verdict"]
            src = m["gps_source"]
            counts["by_source"].setdefault(src, {"kept": 0, "dropped": 0})
            counts["by_verdict"].setdefault(verdict, 0)
            counts["by_verdict"][verdict] += 1

            # 1) GPS source × verdict matrix
            keep = TRUSTED_MATRIX.get(src, {}).get(verdict, False)
            if not keep:
                counts["dropped_matrix"] += 1
                counts["by_source"][src]["dropped"] += 1
                continue

            # 2) heading must be high-confidence
            #    (decision: gps without trustworthy heading can't be turned
            #    into perspective-relative directions)
            hd = heading.get(fid, {})
            if hd.get("heading_confidence") != "high":
                counts["dropped_heading"] += 1
                counts["by_source"][src]["dropped"] += 1
                continue

            # use HMM-snapped GPS if available, else merged GPS
            h = hmm.get(fid)
            if h and h.get("matched"):
                gps = [h["snap_lat"], h["snap_lon"]]
                hmm_perp = h.get("perp_m")
            else:
                gps = m["gps"]
                hmm_perp = None

            row = {
                "frame_id": fid,
                "image": str(P["frames_dir"] / f"{fid}.jpg"),
                "gps": gps,
                "gps_source": src,
                "heading": hd.get("heading"),
                "heading_confidence": hd.get("heading_confidence", "unknown"),
                "evidence": {
                    **m.get("evidence", {}),
                    "verdict": verdict,
                    "verdict_reason": v["evidence"].get("reason"),
                    "vlm_scanned": v.get("vlm_scanned", False),
                    "hmm_perp_m": hmm_perp,
                },
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["kept"] += 1
            counts["by_source"][src]["kept"] += 1

    print(f"[step11:{args.video}] {counts['kept']}/{n_total} kept "
          f"→ {P['step_11_trusted']}")
    print(f"  dropped: no_verdict={counts['dropped_no_verdict']} "
          f"matrix={counts['dropped_matrix']} "
          f"heading={counts['dropped_heading']}")
    print(f"  by_verdict: {counts['by_verdict']}")
    print(f"  by_source:  {counts['by_source']}")


if __name__ == "__main__":
    main()
