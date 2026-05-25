"""GPS-recovery reconciliation (DEV_MANUAL §2.5, Q4).

Combines the two independent GPS estimates for a video frame — the
DINOv2 visual match and the VLM place-naming — into an accept/reject
decision plus a reconciled GPS.

Two reconcilers are exposed:

  reconcile_strict   (DEFAULT, used by src.gps_recovery)
      Three independent filters — any failure rejects:
        1.  cosine >= MIN_SIM             (DINOv2 has a real match)
        2.  VLM gps present (resolved to a known POI)
        3.  |g_dino - g_vlm| <= MAX_VAR_M (they agree spatially)
      Blend: 50/50 midpoint of g_dino and g_vlm. *No single criterion
      can overrule another* — important because the VLM is fallible.

  reconcile_weighted (legacy / ablation)
      Q = w_s·cosine + w_a·agreement + w_c·vlm_confidence, accept if
      Q >= tau. A high cosine or high VLM confidence can drag the
      decision past tau even when the two GPSes disagree by ~200 m —
      that's why the strict variant is the default now.

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


def reconcile_strict(dino_gps, cosine_sim, vlm_gps, vlm_confidence,
                     min_sim: float = config.MIN_SIM,
                     max_var_m: float = config.RECONCILE_MAX_VAR_M,
                     semantic_match: bool = False) -> dict:
    """Strict multi-filter reconcile + 50/50 GPS blend (DEFAULT).

    F1.  cosine_sim >= min_sim         — DINOv2 has a real visual match
    F2.  vlm_gps is not None           — VLM mapped to a known POI
    F3.  semantic_match                — VLM's named place == OSM POI
                                          nearest to g_dino (the only
                                          F3 gate; the spatial fallback
                                          was deliberately dropped to
                                          require true *semantic*
                                          agreement, not just spatial
                                          coincidence)

    `max_var_m` is *not* an accept gate anymore — it only decides how
    to blend GPS on accept (close → midpoint, far → g_dino alone, so
    long-feature centroids don't pollute the result).

    The caller computes `semantic_match` (see src.spatial.nearest_poi_m
    + name comparison against vlm_place_name); reconcile_strict stays
    pure (no POI / file I/O).

    On accept the GPS is the **midpoint of g_dino and g_vlm** (50/50)
    — we deliberately do NOT lean towards either signal because the VLM
    can be confidently wrong. `score` is reported for diagnostics but
    is *not* the accept gate.

    Returns: {accepted, score, variance_m, gps, reject_reason,
              spatial_match}.
    """
    out = {"accepted": False, "score": cosine_sim, "variance_m": None,
           "gps": None, "reject_reason": "",
           "spatial_match": None}
    if cosine_sim < min_sim:
        out["reject_reason"] = "dino_weak"
        return out
    if vlm_gps is None:
        out["reject_reason"] = "vlm_unresolved"
        return out
    d = haversine_m(dino_gps[0], dino_gps[1], vlm_gps[0], vlm_gps[1])
    out["variance_m"] = d
    out["score"] = cosine_sim * agreement(d) * conf_num(vlm_confidence)
    spatial_match = d <= max_var_m
    out["spatial_match"] = spatial_match           # diagnostic only
    if not semantic_match:
        out["reject_reason"] = "disagree"
        return out
    out["accepted"] = True
    if spatial_match:
        # close in metres -> 50/50 midpoint is meaningful
        out["gps"] = ((dino_gps[0] + vlm_gps[0]) / 2,
                      (dino_gps[1] + vlm_gps[1]) / 2)
    else:
        # semantic match but far apart: VLM's resolved POI is a long /
        # large feature (Limmat river, Zürichsee, Bahnhofstrasse) whose
        # *centroid* is far from where the photo was taken. Don't blend
        # with a centroid that would yield a meaningless midpoint —
        # trust DINOv2's coordinates and record the semantic
        # confirmation in `dino_nearest_name` / `place_guess`.
        out["gps"] = (dino_gps[0], dino_gps[1])
    return out


def reconcile_weighted(dino_gps, cosine_sim, vlm_gps, vlm_confidence,
                       tau: float = config.RECONCILE_TAU) -> dict:
    """Legacy weighted-Q reconcile (kept for ablation). See module
    docstring for why strict is the default now."""
    d = haversine_m(dino_gps[0], dino_gps[1], vlm_gps[0], vlm_gps[1])
    score = combined_score(cosine_sim, d, vlm_confidence)
    if score < tau:
        return {"accepted": False, "score": score,
                "variance_m": d, "gps": None}
    cw = conf_num(vlm_confidence)
    total = cosine_sim + cw
    lat = (dino_gps[0] * cosine_sim + vlm_gps[0] * cw) / total
    lon = (dino_gps[1] * cosine_sim + vlm_gps[1] * cw) / total
    return {"accepted": True, "score": score,
            "variance_m": d, "gps": (lat, lon)}


# Backwards-compatible alias. Existing callers get the strict variant.
reconcile = reconcile_strict
