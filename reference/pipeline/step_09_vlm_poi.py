"""Step 9: extract per-video VLM POI scan results.

The actual scan was already run by `toolbox/scan_video_pois_multi.py`,
which writes a single combined `_video_poi_multi.jsonl`. This step
splits that file into per-video step_09 outputs aligned with the
canonical naming.

Run:
    python -m pipeline.step_09_vlm_poi --video bahnhofstrasse
    python -m pipeline.step_09_vlm_poi --video all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import VIDEOS, VIDEO_NAMES, paths_for, VLM_POI_SCAN  # noqa: E402


# Map source-side `video` field (from scan_video_pois_multi.py) to canonical name.
# scan_video_pois_multi strips the 'extra_' prefix already, so source key
# matches our raw_prefix exactly (e.g. 'Switzerland_Zurich_Bahnhofstrasse...').
SOURCE_TO_CANONICAL = {}
for canonical, frames_name, prefix in VIDEOS:
    if prefix:
        SOURCE_TO_CANONICAL[prefix] = canonical
    else:
        # Original = frames/zurich; scan_video_pois_multi labels it 'Original'
        SOURCE_TO_CANONICAL["Original"] = canonical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="", help="output namespace suffix (e.g. _hq)")
    ap.add_argument("--video", required=True,
                    choices=VIDEO_NAMES + ["all"])
    args = ap.parse_args()

    if not VLM_POI_SCAN.exists():
        sys.exit(f"missing VLM POI scan: {VLM_POI_SCAN}")

    targets = VIDEO_NAMES if args.video == "all" else [args.video]

    # Bucket the combined file by canonical video name
    by_video = {v: [] for v in targets}
    for ln in open(VLM_POI_SCAN):
        r = json.loads(ln)
        canon = SOURCE_TO_CANONICAL.get(r.get("video", ""))
        if canon in by_video:
            by_video[canon].append({
                "frame_id": r["frame_id"],
                "visible_pois": r.get("visible_pois", []),
            })

    for v in targets:
        P = paths_for(v, variant=args.variant)
        P["out_dir"].mkdir(parents=True, exist_ok=True)
        with open(P["step_09_vlm_poi"], "w") as fout:
            for r in by_video[v]:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[step9:{v}] wrote {len(by_video[v])} rows → {P['step_09_vlm_poi']}")


if __name__ == "__main__":
    main()
