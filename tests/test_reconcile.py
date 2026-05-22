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
    r = rec.reconcile((47.37, 8.54), 0.9, (47.37, 8.54), "high")
    assert r["accepted"] is True
    assert r["variance_m"] < 1.0
    assert r["gps"] is not None


def test_reconcile_rejects_when_estimates_disagree():
    r = rec.reconcile((47.37, 8.54), 0.15, (47.30, 8.60), "low")
    assert r["accepted"] is False
    assert r["gps"] is None
