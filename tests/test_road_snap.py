"""Unit tests for src/road_snap.py — HMM map-matching core."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import road_snap as rs  # noqa: E402


def test_emission_logp_closer_scores_higher():
    assert rs.emission_logp(0) == 0.0
    assert rs.emission_logp(10) > rs.emission_logp(50)


def test_transition_logp_penalises_detour():
    assert rs.transition_logp(100, 100) == 0.0       # on-road == straight
    assert rs.transition_logp(100, 500) < 0          # big detour penalised


def test_viterbi_empty():
    assert rs.viterbi([], lambda t, s: 0, lambda t, p, c: 0) == []


def test_viterbi_follows_emission():
    obs = [["good", "bad"]] * 3
    emit = lambda t, s: 0.0 if s == "good" else -10.0   # noqa: E731
    trans = lambda t, p, c: 0.0                          # noqa: E731
    assert rs.viterbi(obs, emit, trans) == ["good", "good", "good"]


def test_viterbi_follows_transition():
    obs = [["start"], ["up", "down"], ["end"]]
    emit = lambda t, s: 0.0                              # noqa: E731

    def trans(t, prev, cur):
        return 0.0 if "up" in (prev, cur) else -5.0

    assert rs.viterbi(obs, emit, trans)[1] == "up"
