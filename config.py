"""Central configuration for NavLM v2.

Every path is derived from DATA_ROOT — no hardcoded absolute paths live
in `src/`. Override the data location with the NAVLM_DATA env var.
"""

import os
import shutil
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent

# Raw data + pipeline outputs (local disk, gitignored).
# Default: <repo>/data ; override with `set NAVLM_DATA=...`.
DATA_ROOT = Path(os.environ.get("NAVLM_DATA", REPO_ROOT / "data"))

CITY = "zurich"
CITY_DIR = DATA_ROOT / "cities" / CITY
VIDEOS_DIR = REPO_ROOT / "videos"        # source videos (already present)
FRAMES_DIR = CITY_DIR / "frames"
STREETVIEW_DIR = DATA_ROOT / "cities" / "streetview" / CITY
VIZ_DIR = REPO_ROOT / "viz"

# ── GPS scope ────────────────────────────────────────────────────────
# Central Zurich old town (the OSM POI bbox), as (W, S, E, N).
POI_BBOX = (8.520, 47.360, 8.570, 47.395)
# Street View crawl bbox = POI bbox + ~300 m margin on each side (§3.4),
# so edge POIs and routes that leave the POI box still have imagery.
SV_BBOX = (8.515, 47.355, 8.575, 47.400)

# ── the 8 source videos: youtube_id -> dataset name ──────────────────
VIDEOS = {
    "h7saB68KE5M": "zurich_main",
    "g21yfR4yNd8": "bahnhofstrasse",
    "F8KpE5iEvW0": "most_famous",
    "8zcXNiWRgtA": "saturday_morning",   # evaluation hold-out
    "3BnA_kP2HHY": "looks_perfect",
    "JUuggKe733s": "old_town_limmat",
    "5175ziTF3Gc": "most_elegant",
    "QU1HxFTuqPY": "hidden_streets",
}
HOLDOUT_VIDEO = "saturday_morning"

# Source video files are named by their YouTube title (not id). Map a
# distinctive lowercase filename substring -> dataset name. Checked in
# order; the generic "4k 60fps" fallback (the plain "ZURICH … 4K 60fps"
# video) is last so the more specific titles match first.
VIDEO_KEYWORDS = [
    ("hidden streets",   "hidden_streets"),
    ("bahnhofstrasse",   "bahnhofstrasse"),
    ("most famous",      "most_famous"),
    ("saturday morning", "saturday_morning"),
    ("too perfect",      "looks_perfect"),
    ("old town",         "old_town_limmat"),
    ("most elegant",     "most_elegant"),
    ("4k 60fps",         "zurich_main"),
]


def dataset_name(filename: str) -> str:
    """Map a source video filename to its dataset name (see VIDEO_KEYWORDS).
    Falls back to a sanitized stem when nothing matches."""
    low = filename.lower()
    for keyword, name in VIDEO_KEYWORDS:
        if keyword in low:
            return name
    stem = Path(filename).stem
    return "".join(c if c.isalnum() else "_" for c in stem).strip("_").lower()[:40]

# ── frame extraction ─────────────────────────────────────────────────
DENSE_FPS = 1.0           # ffmpeg dense sampling rate
PHASH_THRESHOLD = 10      # perceptual-hash dedup distance (bits)
BLUR_MIN_VAR = 100.0      # variance-of-Laplacian floor; below = too blurry
EXPOSURE_DARK = 25        # mean luma below this = too dark
EXPOSURE_BRIGHT = 230     # mean luma above this = blown out

# ── Street View reference crawl ──────────────────────────────────────
SV_MARGIN_M = 300.0               # crawl-bbox margin around visited POIs
SV_GRID_M = 50.0                  # metadata grid spacing, metres
SV_HEADINGS = [0, 90, 180, 270]   # crops downloaded per panorama
SV_IMG_SIZE = "640x640"
SV_FOV = 90

# ── GPS recovery / matching ──────────────────────────────────────────
DINOV2_TOPK = 5           # k nearest Street View crops per video frame
MIN_SIM = 0.0             # absolute cosine-similarity floor (tuned later)
RECONCILE_D0_M = 150.0    # agreement-term decay scale (haversine metres)
RECONCILE_WEIGHTS = (0.4, 0.4, 0.2)   # (cosine, agreement, VLM-confidence)
RECONCILE_TAU = 0.5       # accept a frame when combined score >= tau

# ── models ───────────────────────────────────────────────────────────
DINOV2_MODEL = "facebook/dinov2-base"
GEMINI_GEOCHECK = "gemini-2.5-pro"     # VLM geo-localization (Q6: Pro)
GEMINI_ANNOTATE = "gemini-2.5-pro"     # instruction annotation
DEST_PER_FRAME = 3                     # annotation destinations / frame (Q6)


# ── tools ────────────────────────────────────────────────────────────
def _find_ffmpeg() -> str:
    """Locate ffmpeg: $FFMPEG env > PATH > winget install dir > 'ffmpeg'.

    winget installs update PATH only for new shells, so we also probe the
    Gyan.FFmpeg install directory directly.
    """
    env = os.environ.get("FFMPEG")
    if env and Path(env).exists():
        return env
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"
    for exe in base.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"):
        return str(exe)
    return "ffmpeg"   # last resort — assume it is on PATH

FFMPEG = _find_ffmpeg()
FFMPEG_DIR = str(Path(FFMPEG).parent)


def summary():
    """Print the resolved config — handy sanity check."""
    print(f"REPO_ROOT  = {REPO_ROOT}")
    print(f"DATA_ROOT  = {DATA_ROOT}")
    print(f"  videos   -> {VIDEOS_DIR}")
    print(f"  frames   -> {FRAMES_DIR}")
    print(f"POI_BBOX   = {POI_BBOX}")
    print(f"SV_BBOX    = {SV_BBOX}")
    print(f"videos     = {len(VIDEOS)}  (hold-out: {HOLDOUT_VIDEO})")
    print(f"ffmpeg     = {FFMPEG}")


if __name__ == "__main__":
    summary()
