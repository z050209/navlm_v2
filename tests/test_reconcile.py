"""Unit tests for src/reconcile.py — the GPS-recovery weighted score."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import reconcile as rec  # noqa: E402


def test_haversine_zero_and_one_degree():
    assert rec.haversine_m(47.37, 8.54, 47.37, 8.54) == 0.0
    d = rec.haversine_m(47.0, 8.0, 48.0, 8.0)      # ~1° latitude
    assert 110_000 < d < 112_000


def test_agreement_decays_with_distance():
    assert rec.agreement(0.0) == 1.0
    assert rec.agreement(150.0) < rec.agreement(50.0)
    assert 0.0 < rec.agreement(1000.0) < 0.1


def test_conf_num_mapping():
    assert rec.conf_num("high") == 1.0
    assert rec.conf_num("medium") == 0.6
    assert rec.conf_num("low") == 0.3
    assert rec.conf_num("nonsense") == 0.3        # unknown -> low


def test_combined_score_bounded_and_monotonic():
    hi = rec.combined_score(0.9, 10.0, "high")
    lo = rec.combined_score(0.2, 5000.0, "low")
    assert 0.0 <= lo < hi <= 1.0


def test_reconcile_accepts_when_estimates_agree():
    # need semantic_match=True now that F3 is name-only
    r = rec.reconcile((47.37, 8.54), 0.9, (47.37, 8.54), "high",
                      semantic_match=True)
    assert r["accepted"] is True
    assert r["variance_m"] < 1.0
    assert r["gps"] is not None


def test_reconcile_rejects_when_estimates_disagree():
    r = rec.reconcile((47.37, 8.54), 0.15, (47.30, 8.60), "low")
    assert r["accepted"] is False
    assert r["gps"] is None


# ── reconcile_strict (the default; user-driven multi-filter design) ──

def test_strict_accepts_when_all_filters_pass():
    """F1+F2+F3 (semantic) pass and estimates are close (<MAX_VAR_M)
    -> midpoint blend accepted."""
    r = rec.reconcile_strict((47.37, 8.54), 0.80, (47.371, 8.541), "high",
                             min_sim=0.60, max_var_m=150.0,
                             semantic_match=True)
    assert r["accepted"] is True
    # midpoint blend, NOT confidence-weighted
    assert abs(r["gps"][0] - (47.37 + 47.371) / 2) < 1e-9
    assert abs(r["gps"][1] - (8.54 + 8.541) / 2) < 1e-9
    assert r["reject_reason"] == ""


def test_strict_rejects_dino_weak():
    r = rec.reconcile_strict((47.37, 8.54), 0.50, (47.37, 8.54), "high",
                             min_sim=0.60)
    assert r["accepted"] is False
    assert r["reject_reason"] == "dino_weak"
    assert r["gps"] is None


def test_strict_rejects_vlm_unresolved():
    r = rec.reconcile_strict((47.37, 8.54), 0.80, None, "")
    assert r["accepted"] is False
    assert r["reject_reason"] == "vlm_unresolved"
    assert r["gps"] is None


def test_strict_rejects_disagree_even_with_high_vlm_conf():
    """The whole point: a confident VLM cannot override spatial
    disagreement. Estimates ~700m apart -> reject."""
    r = rec.reconcile_strict((47.37, 8.54), 0.80, (47.376, 8.548), "high",
                             min_sim=0.60, max_var_m=150.0)
    assert r["accepted"] is False
    assert r["reject_reason"] == "disagree"
    assert r["variance_m"] > 150.0
    assert r["gps"] is None


def test_strict_blend_is_exact_midpoint():
    """No confidence weighting — the 50/50 was deliberate."""
    r = rec.reconcile_strict((47.0, 8.0), 0.95, (47.001, 8.001), "low",
                             semantic_match=True)
    assert r["accepted"] is True
    assert r["gps"] == ((47.0 + 47.001) / 2, (8.0 + 8.001) / 2)


def test_reconcile_alias_uses_strict():
    """The bare `reconcile()` callable must point at the strict path."""
    assert rec.reconcile is rec.reconcile_strict


# ── F3 semantic OR spatial fallback ───────────────────────────────────

def test_strict_semantic_only_uses_dino_gps_not_centroid():
    """Long features (Limmat, Bahnhofstrasse) have centroids far from
    actual position. Semantic-only accept must NOT midpoint with the
    VLM centroid — it must trust DINOv2's coords. Otherwise the
    accepted GPS is a meaningless midpoint."""
    r = rec.reconcile_strict((47.370, 8.530), 0.80,
                             (47.380, 8.500), "high",   # ~2.6 km away
                             min_sim=0.65, max_var_m=150.0,
                             semantic_match=True)
    assert r["accepted"] is True
    assert r["spatial_match"] is False
    # gps == g_dino, NOT the midpoint
    assert r["gps"] == (47.370, 8.530)


def test_strict_both_match_uses_midpoint():
    """When both spatial AND semantic agree, the 50/50 midpoint is
    correct and informative."""
    r = rec.reconcile_strict((47.370, 8.530), 0.80,
                             (47.371, 8.531), "high",   # ~130 m
                             min_sim=0.65, max_var_m=150.0,
                             semantic_match=True)
    assert r["accepted"] is True
    assert r["spatial_match"] is True
    assert abs(r["gps"][0] - 47.3705) < 1e-6
    assert abs(r["gps"][1] - 8.5305) < 1e-6


def test_strict_no_spatial_fallback_anymore():
    """The 150 m spatial fallback was deliberately dropped. A close
    spatial agreement WITHOUT a name match no longer accepts."""
    r = rec.reconcile_strict((47.370, 8.530), 0.80,
                             (47.3705, 8.5305), "high",
                             min_sim=0.65, max_var_m=150.0,
                             semantic_match=False)
    assert r["accepted"] is False
    assert r["reject_reason"] == "disagree"
    assert r["spatial_match"] is True              # logged but unused


def test_strict_rejects_when_no_semantic_match_no_matter_distance():
    r = rec.reconcile_strict((47.370, 8.530), 0.80,
                             (47.380, 8.560), "high",
                             min_sim=0.65, max_var_m=150.0,
                             semantic_match=False)
    assert r["accepted"] is False
    assert r["reject_reason"] == "disagree"
    assert r["spatial_match"] is False
