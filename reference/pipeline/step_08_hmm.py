"""Step 8: HMM map-matching of merged GPS to OSM walking graph.

Thin wrapper around `toolbox/map_match.py`. The wrapper exists so the
pipeline can take step 7's merged output (single per-frame GPS) and
produce a clean per-frame snap_lat/snap_lon for step 10's verifier.

Run:
    python -m pipeline.step_08_hmm --video bahnhofstrasse
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import VIDEO_NAMES, paths_for, OSM_GRAPH, REPO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, choices=VIDEO_NAMES)
    ap.add_argument("--variant", default="", help="output namespace suffix (e.g. _hq)")
    ap.add_argument("--sigma", type=float, default=20.0)
    ap.add_argument("--beta",  type=float, default=10.0)
    ap.add_argument("--max-radius-m", type=float, default=80.0)
    args = ap.parse_args()

    P = paths_for(args.video, variant=args.variant)
    if not P["step_07_merged"].exists():
        sys.exit(f"missing step 7 output: {P['step_07_merged']}")

    cmd = [
        sys.executable, str(REPO / "toolbox" / "map_match.py"),
        "--input", str(P["step_07_merged"]),
        "--graph", str(OSM_GRAPH),
        "--out",   str(P["step_08_hmm"]),
        "--sigma", str(args.sigma),
        "--beta",  str(args.beta),
        "--max-radius-m", str(args.max_radius_m),
    ]
    print(f"[step8:{args.video}] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # quick stats
    matched = total = 0
    perp_sum = 0.0
    for ln in open(P["step_08_hmm"]):
        r = json.loads(ln)
        total += 1
        if r.get("matched"):
            matched += 1
            perp_sum += r.get("perp_m", 0.0)
    if matched:
        print(f"[step8:{args.video}] matched={matched}/{total} "
              f"({100*matched/total:.1f}%) mean_perp={perp_sum/matched:.1f}m")


if __name__ == "__main__":
    main()
