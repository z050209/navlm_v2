"""Unit tests for src/annotate.py — destination sampling + verifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import annotate as an  # noqa: E402


def test_sample_destinations_count_and_unique():
    cands = [(f"p{i}", 100 + i * 100) for i in range(15)]   # 100..1500 m
    out = an.sample_destinations(cands, n=3, seed=1)
    assert len(out) == 3
    assert len({c[0] for c in out}) == 3                    # no duplicates


def test_sample_destinations_caps_at_candidates():
    out = an.sample_destinations([("a", 100), ("b", 300)], n=3, seed=1)
    assert len(out) == 2


def test_sample_destinations_band_bias():
    cands = ([(f"near{i}", 100 + i * 30) for i in range(12)]    # < 500 m
             + [(f"mid{i}", 600 + i * 30) for i in range(12)]   # 500-1000
             + [(f"far{i}", 1100 + i * 30) for i in range(12)]) # 1000-1500
    near = total = 0
    for s in range(80):
        for _, d in an.sample_destinations(cands, n=3, seed=s):
            total += 1
            near += d < 500
    assert near / total > 0.6        # biased toward the ≤500 m band


def test_verify():
    assert an.verify(0, "continue ahead", 0) is True
    assert an.verify(0, "turn left", 90) is False    # wrong way -> fails
