"""Unit tests for src/routing.py — bearing / action geometry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import routing as rt  # noqa: E402


def test_bearing_cardinal_directions():
    assert abs(rt.bearing_deg(47.0, 8.0, 47.1, 8.0) - 0) < 1    # north
    assert abs(rt.bearing_deg(47.0, 8.0, 47.0, 8.1) - 90) < 1   # east


def test_angle_diff_wraps():
    assert rt.angle_diff(10, 350) == 20
    assert rt.angle_diff(350, 10) == -20
    assert rt.angle_diff(0, 0) == 0


def test_action_for():
    assert rt.action_for(0) == "continue ahead"
    assert rt.action_for(20) == "continue ahead"
    assert rt.action_for(-90) == "turn left"
    assert rt.action_for(90) == "turn right"
    assert rt.action_for(180) == "turn around"
    assert rt.action_for(-170) == "turn around"


def test_closed_loop_delta_zero_when_consistent():
    assert rt.closed_loop_delta(0, "continue ahead", 0) == 0
    assert rt.closed_loop_delta(0, "turn right", 90) == 0
    assert rt.closed_loop_delta(0, "turn around", 180) == 0


def test_closed_loop_delta_flags_wrong_action():
    # facing 0, route is 90° (a right turn) but the action says "left"
    assert rt.closed_loop_delta(0, "turn left", 90) > 90


def test_distance_phrase_bands():
    assert rt.distance_phrase(10) == "just a few steps"
    assert rt.distance_phrase(400) == "a few blocks"
    assert rt.distance_phrase(2000) == "several blocks"


def test_action_delta_keys():
    assert set(rt.ACTION_DELTA) == {
        "continue ahead", "turn left", "turn right", "turn around"}
