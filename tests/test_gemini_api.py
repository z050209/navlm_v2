"""Unit tests for src/gemini_api.py — pure helpers (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import gemini_api as g  # noqa: E402


def test_cost_usd_flash():
    # flash list price: 0.30 in / 2.50 out per 1M tokens
    c = g.cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000)
    assert abs(c - (0.30 + 2.50)) < 1e-9
    assert g.cost_usd("gemini-2.5-flash", 0, 0) == 0.0


def test_cost_usd_pro_dearer_than_flash():
    pro = g.cost_usd("gemini-2.5-pro", 500_000, 500_000)
    flash = g.cost_usd("gemini-2.5-flash", 500_000, 500_000)
    assert pro > flash


def test_retry_delay_parses_retryinfo():
    body = {"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.Help"},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo",
         "retryDelay": "26s"}]}}
    assert g.retry_delay_s(body) == 26.0


def test_retry_delay_default_when_absent():
    assert g.retry_delay_s({}, default=12.0) == 12.0
    assert g.retry_delay_s({"error": {"details": []}}, default=7.0) == 7.0


def test_vertex_url_global():
    u = g.vertex_url("gemini-2.5-pro", "myproj", "global")
    assert u.startswith(
        "https://aiplatform.googleapis.com/v1/projects/myproj"
        "/locations/global/")
    assert u.endswith("publishers/google/models/"
                      "gemini-2.5-pro:generateContent")


def test_vertex_url_regional():
    u = g.vertex_url("gemini-2.5-pro", "myproj", "us-central1")
    assert u.startswith("https://us-central1-aiplatform.googleapis.com/")
    assert "/locations/us-central1/" in u
