"""Stage 2 — video -> frames, with a quality filter (new in v2).

Pipeline per video:
  1. ffmpeg dense-sample at DENSE_FPS into <name>_dense/   (cache, kept)
  2. quality gate — drop blurry (variance-of-Laplacian) and badly
     exposed (mean-luma) frames
  3. perceptual-hash dedup — keep a frame only if its pHash differs from
     the last *kept* frame by >= PHASH_THRESHOLD bits
  -> <DATA_ROOT>/cities/zurich/frames/<name>/frame_NNNNN.jpg

v1 (`reference/toolbox/extract_frames.py`) did steps 1+3 only — no
quality filter, so blurry frames passed. Step 2 is the v2 addition.

    python -m src.extract_frames                  # all downloaded videos
    python -m src.extract_frames --only saturday_morning

Requires: ffmpeg on PATH, and `pip install imagehash pillow opencv-python`.
Thresholds (DENSE_FPS, PHASH_THRESHOLD, BLUR_MIN_VAR, EXPOSURE_*) live
in config.py.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def dense_sample(video: Path, dense_dir: Path) -> list:
    """ffmpeg dense-sample at config.DENSE_FPS. Cached — reused if present."""
    existing = sorted(dense_dir.glob("*.jpg"))
    if existing:
        print(f"    reusing {len(existing)} cached dense frames")
        return existing
    dense_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={config.DENSE_FPS}", "-q:v", "3",
         str(dense_dir / "dense_%06d.jpg")],
        check=True,
    )
    return sorted(dense_dir.glob("*.jpg"))


def quality_metrics(path: Path):
    """Return (laplacian_variance, mean_luma) for a frame."""
    import cv2
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0.0, 0.0
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(blur), float(gray.mean())


def extract_video(video: Path, name: str) -> dict:
    """Extract one video -> quality-filtered, deduped frames."""
    import imagehash
    from PIL import Image

    out_dir = config.FRAMES_DIR / name
    dense_dir = config.FRAMES_DIR / f"{name}_dense"
    print(f"  {name}")
    dense = dense_sample(video, dense_dir)
    print(f"    {len(dense)} dense frames @ {config.DENSE_FPS} fps")

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    kept = n_blur = n_exposure = n_dup = 0
    last_hash = None
    for f in dense:
        blur, luma = quality_metrics(f)
        if blur < config.BLUR_MIN_VAR:
            n_blur += 1
            continue
        if luma < config.EXPOSURE_DARK or luma > config.EXPOSURE_BRIGHT:
            n_exposure += 1
            continue
        h = imagehash.phash(Image.open(f))
        if last_hash is not None and (h - last_hash) < config.PHASH_THRESHOLD:
            n_dup += 1
            continue
        (out_dir / f"frame_{kept:05d}.jpg").write_bytes(f.read_bytes())
        kept += 1
        last_hash = h

    stats = {"video": name, "dense": len(dense), "kept": kept,
             "dropped_blur": n_blur, "dropped_exposure": n_exposure,
             "dropped_duplicate": n_dup}
    print(f"    kept {kept}  "
          f"(dropped: blur={n_blur} exposure={n_exposure} dup={n_dup})")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="dataset name (default: all downloaded)")
    args = ap.parse_args()

    videos = {name: config.VIDEOS_DIR / f"{name}.mp4"
              for name in config.VIDEOS.values()}
    todo = [(n, p) for n, p in videos.items()
            if p.exists() and (not args.only or n == args.only)]
    if not todo:
        sys.exit(f"no videos found in {config.VIDEOS_DIR} — "
                 f"run `python -m src.download_videos` first")

    print(f"extracting {len(todo)} video(s) -> {config.FRAMES_DIR}\n")
    all_stats = [extract_video(p, n) for n, p in todo]

    total = sum(s["kept"] for s in all_stats)
    report = config.FRAMES_DIR / "extract_report.json"
    report.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\ntotal kept frames: {total}")
    print(f"per-video report -> {report}")


if __name__ == "__main__":
    main()
