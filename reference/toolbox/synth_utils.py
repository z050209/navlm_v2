"""Shared geometry + VLM-call utilities for training-data synthesis.

Used by:
  - synth_algo1_video_routes.py   (video-traversal-derived directions)
  - synth_algo2_visible_scene.py  (current-frame scene grounding)
"""

import base64
import json
import math
import re
from pathlib import Path

import requests


# ---------- geometry ----------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing 0=N 90=E from (lat1,lon1) to (lat2,lon2)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def relative_to_facing(heading_deg, target_bearing_deg):
    """Translate compass bearing to user-perspective direction word.

    Returns one of: 'directly ahead', 'ahead and slightly right',
    'to your right', 'behind and to your right', 'directly behind you',
    'behind and to your left', 'to your left', 'ahead and slightly left'.
    """
    diff = (target_bearing_deg - heading_deg + 360) % 360
    if diff < 22.5 or diff >= 337.5:
        return "directly ahead"
    if diff < 67.5:
        return "ahead and to your right"
    if diff < 112.5:
        return "to your right"
    if diff < 157.5:
        return "behind you and to your right"
    if diff < 202.5:
        return "directly behind you"
    if diff < 247.5:
        return "behind you and to your left"
    if diff < 292.5:
        return "to your left"
    return "ahead and to your left"


def estimate_heading(frames_sorted, idx, max_window=10, min_dist_m=5.0):
    """Estimate heading at frame idx from nearest high-conf forward neighbor.

    frames_sorted: list of dicts with keys frame_id, gps, confidence (sorted by index).
    Returns heading in degrees or None.
    """
    cur = frames_sorted[idx]
    if cur.get("confidence") not in ("high", "medium"):
        return None
    for k in range(1, max_window + 1):
        if idx + k >= len(frames_sorted):
            break
        nxt = frames_sorted[idx + k]
        if nxt.get("confidence") not in ("high", "medium"):
            continue
        d = haversine_m(cur["gps"][0], cur["gps"][1], nxt["gps"][0], nxt["gps"][1])
        if d >= min_dist_m:
            return bearing_deg(cur["gps"][0], cur["gps"][1], nxt["gps"][0], nxt["gps"][1])
    return None


# ---------- VLM ----------

def img_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    ext = path.suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def call_vlm(image_paths, text_prompt, vllm_url, model, max_tokens=400,
             temperature=0.2, timeout=120):
    """Call vLLM /chat/completions with one or more images + a text prompt."""
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    content = [{"type": "text", "text": text_prompt}]
    for p in image_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": img_to_data_url(Path(p))}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{vllm_url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------- scrub: enforce no-leakage policy ----------

# Case-insensitive: full compass words + units + GPS lingo.
_FORBIDDEN_CI = re.compile(
    r"\b(north|south|east|west|northeast|northwest|southeast|southwest)(ern|ward|wards)?\b"
    r"|\b\d{1,5}\s*(meter|metre|meters|metres|km|kilometer|kilometre)\b"
    r"|\b\d{1,5}\s?m\b"
    r"|\bGPS\b|\blatitude\b|\blongitude\b|\bcoordinates?\b"
    r"|\bdegrees?\b"
    r"|\b\d+\s*°",
    flags=re.IGNORECASE,
)
# Case-SENSITIVE: bare N/S/E/W and NE/NW/SE/SW only as uppercase
# (avoids matching the 's' in "That's" or other lowercase letters in words).
_FORBIDDEN_CS = re.compile(
    r"\b[NSEW]\b|\bN[EW]\b|\bS[EW]\b"
)


def has_forbidden_terms(text: str):
    """Return list of forbidden phrases found in `text` (empty list = clean)."""
    hits = [m.group(0) for m in _FORBIDDEN_CI.finditer(text)]
    hits += [m.group(0) for m in _FORBIDDEN_CS.finditer(text)]
    return hits


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def frame_idx_from_id(fid: str) -> int:
    """frame_00123 → 123."""
    m = re.search(r"(\d+)", fid)
    return int(m.group(1)) if m else -1
