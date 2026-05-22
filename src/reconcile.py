"""GPS-recovery reconciliation (DEV_MANUAL §2.5, Q4).

Combines the two independent GPS estimates for a video frame —
the DINOv2 visual match and the VLM place-naming — into one weighted
quality score and an accept/reject decision.

    Q = w_s·cosine + w_a·agreement + w_c·vlm_confidence
    agreement = exp(-haversine(dino_gps, vlm_gps) / D0)

All pure functions — fully unit-tested, no network / GPU.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

CONF_NUM = {"high": 1.0, "medium": 0.6, "low": 0.3}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def agreement(distance_m: float, d0: float = config.RECONCILE_D0_M) -> float:
    """exp(-d/D0): 1.0 when the two GPS estimates coincide, decaying."""
    return math.exp(-distance_m / d0)


def conf_num(confidence) -> float:
    """VLM confidence label -> number; unknown -> low."""
    return CONF_NUM.get(str(confidence).lower(), 0.3)


def combined_score(cosine_sim, distance_m, vlm_confidence,
                   weights=config.RECONCILE_WEIGHTS,
                   d0: float = config.RECONCILE_D0_M) -> float:
    """Weighted quality score Q in [0, 1] (DEV_MANUAL §2.5)."""
    w_s, w_a, w_c = weights
    return (w_s * cosine_sim
            + w_a * agreement(distance_m, d0)
            + w_c * conf_num(vlm_confidence))


def reconcile(dino_gps, cosine_sim, vlm_gps, vlm_confidence,
              tau: float = config.RECONCILE_TAU) -> dict:
    """Reconcile the two estimates for one frame.

    dino_gps, vlm_gps: (lat, lon). Returns a dict with:
      accepted   — bool, score >= tau
      score      — the combined quality Q
      variance_m — haversine disagreement of the two estimates
      gps        — accepted (lat, lon), a confidence-weighted blend, or None
    """
    d = haversine_m(dino_gps[0], dino_gps[1], vlm_gps[0], vlm_gps[1])
    score = combined_score(cosine_sim, d, vlm_confidence)
    if score < tau:
        return {"accepted": False, "score": score,
                "variance_m": d, "gps": None}
    # confidence-weighted blend of the two estimates
    cw = conf_num(vlm_confidence)
    total = cosine_sim + cw
    lat = (dino_gps[0] * cosine_sim + vlm_gps[0] * cw) / total
    lon = (dino_gps[1] * cosine_sim + vlm_gps[1] * cw) / total
    return {"accepted": True, "score": score,
            "variance_m": d, "gps": (lat, lon)}
