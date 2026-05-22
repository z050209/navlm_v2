"""Run scene-adaptive frame extraction on the new Zurich_extra videos.

Each video gets its own output dir under data/cities/zurich/frames/
so it doesn't overwrite the original zurich/ 2224 frames.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_frames import dedup_scene_change  # noqa


VIDEOS_DIR = Path("/pub/evaluation_group/ning/test/navlm/data/cities/zurich/videos/Zurich_extra")
OUT_ROOT = Path("/pub/evaluation_group/ning/test/navlm/data/cities/zurich/frames")
DENSE_FPS = 1.0
PHASH_THRESHOLD = 10


def sanitize(name: str) -> str:
    """Make a filesystem-safe short directory name."""
    out = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:50]


def main():
    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no videos in {VIDEOS_DIR}")
    print(f"[extra] {len(videos)} videos to process")

    for i, v in enumerate(videos, 1):
        stem = sanitize(v.stem)
        out_dir = OUT_ROOT / f"extra_{stem}"
        print(f"\n=== [{i}/{len(videos)}] {v.name}")
        print(f"     → {out_dir}")
        # Skip if already done (resume support)
        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0:
            n = len(list(out_dir.glob("*.jpg")))
            print(f"     SKIP — already has {n} frames")
            continue
        try:
            dedup_scene_change(v, out_dir,
                               dense_fps=DENSE_FPS,
                               phash_threshold=PHASH_THRESHOLD)
        except Exception as e:
            print(f"     ERROR: {type(e).__name__}: {str(e)[:200]}")
            continue

    # Final tally
    print("\n=== DONE ===")
    for v in videos:
        stem = sanitize(v.stem)
        out_dir = OUT_ROOT / f"extra_{stem}"
        n = len(list(out_dir.glob("*.jpg"))) if out_dir.exists() else 0
        print(f"  extra_{stem:<50}  {n} frames")


if __name__ == "__main__":
    main()
