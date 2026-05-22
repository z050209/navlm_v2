"""Unit tests for src/gps_recovery.py — matching + heading geometry."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import gps_recovery as gr  # noqa: E402


def _norm(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_cosine_topk_orders_by_similarity():
    q = _norm([1, 0, 0])
    refs = np.stack([_norm([1, 0, 0]), _norm([0, 1, 0]), _norm([0.9, 0.1, 0])])
    idx, sims = gr.cosine_topk(q, refs, k=3)
    assert idx[0] == 0                       # identical vector ranks first
    assert sims[0] >= sims[1] >= sims[2]


def test_cosine_topk_respects_k():
    q = _norm([1, 0, 0])
    refs = np.stack([_norm([1, 0, 0])] * 5)
    idx, _ = gr.cosine_topk(q, refs, k=2)
    assert len(idx) == 2


def test_circular_mean():
    assert abs(gr.circular_mean([10, 20, 30]) - 20) < 1e-6
    m = gr.circular_mean([350, 10])          # wraps around 0
    assert m < 1 or m > 359
    assert gr.circular_mean([]) is None


def test_circular_spread():
    assert gr.circular_spread([90, 90, 90]) == 0.0
    assert gr.circular_spread([0, 90, 180, 270]) > 90   # fully dispersed
    assert gr.circular_spread([]) == 360.0
