"""Phase B — instruction-tuning annotation with Gemini 2.5 Pro.

For each trusted frame: sample destinations, plan the route, ask the
teacher VLM for a scene-anchored answer, and gate it with the
closed-loop verifier.

  sample_destinations()  — 3 destinations / frame by distance band
                           (DEV_MANUAL §2.7 Q6: ≤500m 80%, 500-1000m 10%,
                           1000-1500m 10%)
  verify()               — closed-loop angular check, δ < 30°
  annotate_frame()       — the Gemini 2.5 Pro teacher call

Pure functions (sample_destinations, verify) are unit-tested; the
teacher call needs a GEMINI_API_KEY.

    python -m src.annotate --limit 5      # 5-sample trial first
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402
from src.routing import closed_loop_delta      # noqa: E402

# (min_m, max_m, share) — DEV_MANUAL §2.7 Q6
DIST_BANDS = [(0, 500, 0.80), (500, 1000, 0.10), (1000, 1500, 0.10)]


def sample_destinations(candidates, n=config.DEST_PER_FRAME, seed=0):
    """Pick `n` destinations by distance band.

    candidates: list of (name, distance_m). Returns a list of (name,
    distance_m). Each slot draws a band by its share, then a candidate
    inside that band; empty bands fall back to the nearest unused
    candidate so we always return `n` when enough candidates exist.
    """
    rng = random.Random(seed)
    by_band = []
    for lo, hi, _ in DIST_BANDS:
        by_band.append([c for c in candidates if lo <= c[1] < hi])
    weights = [share for _, _, share in DIST_BANDS]

    chosen, used = [], set()
    pool = sorted(candidates, key=lambda c: c[1])
    for _ in range(n):
        band = rng.choices(range(len(DIST_BANDS)), weights=weights)[0]
        opts = [c for c in by_band[band] if c[0] not in used]
        if not opts:                                   # band empty -> fallback
            opts = [c for c in pool if c[0] not in used]
        if not opts:
            break
        pick = rng.choice(opts)
        chosen.append(pick)
        used.add(pick[0])
    return chosen


def verify(heading, action, route_bearing, max_delta=30.0):
    """Closed-loop verifier — True if the action points the right way
    (DEV_MANUAL §2.7 Q5: the 30° tolerance)."""
    return closed_loop_delta(heading, action, route_bearing) < max_delta


def annotate_frame(image_path, sys_prompt, user_msg):
    """One teacher call to Gemini 2.5 Pro. Returns the raw response."""
    sys.path.insert(0, str(config.REPO_ROOT / "reference" / "toolbox"))
    from synth.backends import call_gemini
    return call_gemini(image_path, sys_prompt, user_msg,
                       model=config.GEMINI_ANNOTATE)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Phase B — annotation")
    ap.add_argument("--limit", type=int, default=5,
                    help="annotate only N frames (5-sample trial first)")
    args = ap.parse_args()
    print(f"[annotate] would annotate {args.limit} frames with "
          f"{config.GEMINI_ANNOTATE}, {config.DEST_PER_FRAME} dest/frame.")
    print("[annotate] wiring to trusted frames + routes is pending the "
          "GPS-recovery stage; run with --limit once those exist.")


if __name__ == "__main__":
    main()
