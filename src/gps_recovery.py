"""Stage 6 — GPS recovery for video frames (DEV_MANUAL §2.5).

For each video frame:
  1. DINOv2 embed -> cosine-match against the Street View index (top-k)
  2. VLM (Gemini Pro) names the place -> resolved to GPS
  3. src.reconcile combines the two into an accept/reject + GPS
  4. heading = circular mean of the matched crops' rendered headings

Pure functions (cosine_topk, circular_mean) are unit-tested; embedding
needs DINOv2/GPU and the VLM step needs a GEMINI_API_KEY.

    python -m src.gps_recovery        # runs the full stage (heavy)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                       # noqa: E402
from src import reconcile           # noqa: E402


def cosine_topk(query_emb, ref_embs, k=config.DINOV2_TOPK):
    """Top-k reference indices + cosine similarities for one query.

    query_emb: (D,) ; ref_embs: (N, D). Both assumed L2-normalised, so
    cosine = dot product. Returns (indices, sims), best-first.
    """
    import numpy as np
    sims = ref_embs @ query_emb
    order = np.argsort(-sims)[:k]
    return order, sims[order]


def circular_mean(degrees):
    """Circular mean of a list of headings (degrees), in [0, 360)."""
    if not degrees:
        return None
    x = sum(math.sin(math.radians(d)) for d in degrees)
    y = sum(math.cos(math.radians(d)) for d in degrees)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def circular_spread(degrees):
    """Circular standard deviation (degrees) — heading confidence proxy."""
    if not degrees:
        return 360.0
    n = len(degrees)
    x = sum(math.sin(math.radians(d)) for d in degrees) / n
    y = sum(math.cos(math.radians(d)) for d in degrees) / n
    r = math.sqrt(x * x + y * y)
    if r >= 0.9999:
        return 0.0
    if r < 1e-9:                       # fully dispersed -> max spread
        return 360.0
    return math.degrees(math.sqrt(-2 * math.log(r)))


def recover_frame(query_emb, ref):
    """Recover (gps, heading) for one frame.

    ref: dict with embs (N,D), gps [(lat,lon)...], headings [deg...],
         and `vlm` = a callable frame->(place_gps, confidence).
    Returns a reconcile result dict + heading fields.
    """
    idx, sims = cosine_topk(query_emb, ref["embs"])
    dino_gps = ref["gps"][int(idx[0])]
    heading = circular_mean([ref["headings"][int(i)] for i in idx])
    spread = circular_spread([ref["headings"][int(i)] for i in idx])

    vlm_gps, vlm_conf = ref["vlm"]()           # VLM place-naming
    result = reconcile.reconcile(dino_gps, float(sims[0]), vlm_gps, vlm_conf)
    result["heading"] = heading
    result["heading_spread"] = spread
    return result


def main():
    print("[gps_recovery] orchestrates DINOv2 match + VLM place-naming + "
          "reconcile + heading.")
    print("[gps_recovery] needs the Street View index (src.streetview) and "
          "extracted frames (src.extract_frames) first.")


if __name__ == "__main__":
    main()
