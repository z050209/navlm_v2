"""Driver: run pipeline steps across all (or one) video.

Examples:
    python -m pipeline.run_all                     # all steps 7-11, all 8 videos
    python -m pipeline.run_all --from-step 8       # skip merge_gps, start at HMM
    python -m pipeline.run_all --videos bahnhofstrasse most_famous
    python -m pipeline.run_all --steps 10 11       # only verify + trusted

After the per-video phase finishes, the driver concatenates all
step_11_trusted.jsonl files into `data/cities/zurich/frame_starts_trusted_all.jsonl`
(the canonical input for synth_unified.py).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import (  # noqa: E402
    VIDEO_NAMES, VIDEOS, paths_for, trusted_starts_path,
    REPO, DATA, LOGS,
)


STEP_MODULES = {
    7:  "pipeline.step_07_merge_gps",
    8:  "pipeline.step_08_hmm",
    9:  "pipeline.step_09_vlm_poi",
    10: "pipeline.step_10_vlm_verify",
    11: "pipeline.step_11_trusted",
}


def run_step(step, video, variant="", no_bbox=False):
    """Invoke a step module via `python -m`. Captures stdout/stderr to log."""
    LOGS.mkdir(parents=True, exist_ok=True)
    tag = variant.lstrip("_") or "default"
    log_path = LOGS / f"{video}_step_{step:02d}_{tag}.log"
    cmd = [sys.executable, "-m", STEP_MODULES[step], "--video", video]
    if variant:
        cmd += ["--variant", variant]
    if no_bbox and step == 7:
        cmd += ["--no-bbox"]
    print(f"  $ {' '.join(cmd)}    > {log_path.relative_to(REPO)}")
    t0 = time.time()
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                           cwd=REPO)
    dt = time.time() - t0
    ok = (r.returncode == 0)
    tag = "OK" if ok else f"FAIL(rc={r.returncode})"
    print(f"     {tag}  {dt:.1f}s")
    return ok


def concat_trusted(videos, out_path, variant=""):
    n_total = 0
    by_video = {}
    by_source = {}
    by_verdict = {}
    with open(out_path, "w") as fout:
        for v in videos:
            P = paths_for(v, variant=variant)
            if not P["step_11_trusted"].exists():
                continue
            n = 0
            for ln in open(P["step_11_trusted"]):
                r = json.loads(ln)
                r["video"] = v
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                src = r["gps_source"]
                vd = r["evidence"].get("verdict", "?")
                by_source[src] = by_source.get(src, 0) + 1
                by_verdict[vd] = by_verdict.get(vd, 0) + 1
            by_video[v] = n
            n_total += n
    print(f"\n=== combined trusted_starts: {n_total} frames → {out_path}")
    for v, n in by_video.items():
        print(f"  {v:20s}  {n}")
    print(f"  by source:  {by_source}")
    print(f"  by verdict: {by_verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-step", type=int, default=7,
                    help="start at this step (inclusive)")
    ap.add_argument("--steps", type=int, nargs="+", default=None,
                    help="explicit list of steps to run (overrides --from-step)")
    ap.add_argument("--videos", nargs="+", default=None,
                    help="subset of videos (default: all)")
    ap.add_argument("--variant", default="",
                    help="output namespace suffix (e.g. '_hq' for HQ Mapillary)")
    ap.add_argument("--no-bbox", action="store_true",
                    help="disable step 7's 1.5km old-town bbox filter")
    args = ap.parse_args()

    videos = args.videos or VIDEO_NAMES
    bad = [v for v in videos if v not in VIDEO_NAMES]
    if bad:
        sys.exit(f"unknown video(s): {bad}\nKnown: {VIDEO_NAMES}")

    if args.steps:
        steps = sorted(set(args.steps))
    else:
        steps = list(range(args.from_step, 12))

    variant = args.variant or ""
    print(f"=== pipeline.run_all ===")
    print(f"  videos:  {videos}")
    print(f"  steps :  {steps}")
    print(f"  variant: {variant or '(default)'}")
    print()

    t0 = time.time()
    failed = []
    for step in steps:
        if step not in STEP_MODULES:
            print(f"-- skip unknown step {step}")
            continue
        # step 9 has --video all support; use it once
        if step == 9:
            print(f"-- step {step}: split combined VLM scan → per-video files")
            cmd = [sys.executable, "-m", STEP_MODULES[9], "--video", "all"]
            if variant:
                cmd += ["--variant", variant]
            log_path = LOGS / f"all_step_09{variant}.log"
            with open(log_path, "w") as logf:
                r = subprocess.run(cmd, stdout=logf,
                                   stderr=subprocess.STDOUT, cwd=REPO)
            tag = "OK" if r.returncode == 0 else "FAIL"
            print(f"     {tag}  log → {log_path.relative_to(REPO)}")
            continue
        print(f"-- step {step} ({STEP_MODULES[step]}) over {len(videos)} video(s)")
        for v in videos:
            ok = run_step(step, v, variant=variant, no_bbox=args.no_bbox)
            if not ok:
                failed.append((step, v))

    # combine final outputs
    out_combined = trusted_starts_path(variant)
    concat_trusted(videos, out_combined, variant=variant)

    dt = time.time() - t0
    print(f"\n=== done in {dt/60:.1f} min  failed={len(failed)} ===")
    if failed:
        for s, v in failed:
            print(f"  step {s}  {v}")


if __name__ == "__main__":
    main()
