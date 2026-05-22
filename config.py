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
VIDEOS_DIR = CITY_DIR / "videos"
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

# ── frame extraction ─────────────────────────────────────────────────
DENSE_FPS = 1.0           # ffmpeg dense sampling rate
PHASH_THRESHOLD = 10      # perceptual-hash dedup distance (bits)
BLUR_MIN_VAR = 100.0      # variance-of-Laplacian floor; below = too blurry
EXPOSURE_DARK = 25        # mean luma below this = too dark
EXPOSURE_BRIGHT = 230     # mean luma above this = blown out

# ── models ───────────────────────────────────────────────────────────
DINOV2_MODEL = "facebook/dinov2-base"
GEMINI_GEOCHECK = "gemini-2.5-flash"   # cheap — VLM geo-localization
GEMINI_ANNOTATE = "gemini-2.5-pro"     # quality — instruction annotation


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
