"""Extract frames from city videos at a configurable FPS (with optional scene dedup).

Outputs into data/cities/{city}/frames/{video_id}/*.jpg

Two modes:
  - Default: fixed-rate ffmpeg sampling (--fps controls the rate)
  - --dedup: sample at 1 fps, then keep only frames whose perceptual hash
    differs enough from the previous kept frame. Gives one frame per
    *scene* instead of one frame per N seconds. Recommended for walking
    tours where the content actually varies (long straight stretches
    produce few keepers; corners and neighborhood changes produce many).

If --static-image-dir is supplied instead, frames are taken directly
from that directory (no ffmpeg needed). Useful for MVP when no video
is available.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/pub/evaluation_group/ning/test/navlm")


def extract_with_ffmpeg(video_path: Path, out_dir: Path, fps: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "3",
        str(out_dir / "frame_%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    n = len(list(out_dir.glob("*.jpg")))
    print(f"  {video_path.name} → {n} frames")
    return n


def dedup_scene_change(video_path: Path, out_dir: Path, dense_fps: float = 1.0,
                       phash_threshold: int = 10, redo_only: bool = False):
    """Sample densely, then keep only frames that differ by >= threshold bits of pHash.

    phash distance meaning (on 64-bit phash):
      <  6 : near-duplicate  (skip)
      6-14: noticeable change (keep)
      > 14: very different  (definitely keep)

    If redo_only=True, skip the ffmpeg dense-sampling step and just re-run
    the pHash filter on the existing dense_dir. This lets you tune the
    threshold in seconds instead of re-decoding the whole video.

    The intermediate dense_dir is KEPT after this function returns so that
    re-thresholding is cheap.
    """
    from PIL import Image
    import imagehash

    dense_dir = out_dir.parent / (out_dir.name + "_dense")

    # Step 1: dense sampling (skip if reusing existing cache)
    if redo_only:
        if not dense_dir.exists():
            raise SystemExit(f"redo_only=True but {dense_dir} doesn't exist")
        print(f"  [redo] reusing dense cache at {dense_dir}")
    else:
        if dense_dir.exists():
            shutil.rmtree(dense_dir)
        dense_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", f"fps={dense_fps}",
            "-q:v", "3",
            str(dense_dir / "dense_%06d.jpg"),
        ]
        subprocess.run(cmd, check=True)

    dense_frames = sorted(dense_dir.glob("*.jpg"))
    print(f"  {video_path.name}: {len(dense_frames)} dense frames @ {dense_fps} fps")

    # Step 2: compute consecutive-frame pHash distances + distribution
    hashes = [imagehash.phash(Image.open(f)) for f in dense_frames]
    dists = [hashes[i] - hashes[i - 1] for i in range(1, len(hashes))]
    if dists:
        bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 999)]
        print("  pHash distance distribution (adjacent frames):")
        for lo, hi in bins:
            n = sum(1 for d in dists if lo <= d < hi)
            bar = "#" * min(40, n * 40 // max(1, len(dists)))
            tag = f"{lo}-{hi-1}" if hi < 999 else f">= {lo}"
            print(f"    {tag:<6} : {bar:<42} {n:5d} ({100 * n / len(dists):4.1f}%)")

    # Step 3: greedy keep — first frame always kept, then keep any frame
    # whose pHash differs from the last kept by >= threshold.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_hash = None
    kept = 0
    for f, h in zip(dense_frames, hashes):
        dist = 999 if last_hash is None else (h - last_hash)
        if dist >= phash_threshold:
            target = out_dir / f"frame_{kept:05d}.jpg"
            shutil.copy2(f, target)
            kept += 1
            last_hash = h
    print(f"  → kept {kept} unique-scene frames (phash threshold={phash_threshold})")
    print(f"  [cache] dense frames retained at {dense_dir} (re-run with --redo-dedup to try a different threshold)")
    return kept


def ingest_static_dir(src_dir: Path, dest: Path, max_images: int | None):
    dest.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in src_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if max_images:
        imgs = imgs[:max_images]
    for i, img in enumerate(imgs):
        target = dest / f"frame_{i:05d}{img.suffix.lower()}"
        if not target.exists():
            os.symlink(img.resolve(), target)
    print(f"  static ingest: {len(imgs)} images → {dest}")
    return len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--fps", type=float, default=0.1,
                    help="frames per second for fixed-rate mode (0.1 = one every 10s)")
    ap.add_argument("--dedup", action="store_true",
                    help="scene-adaptive mode: sample 1fps then keep only frames that look different")
    ap.add_argument("--dense-fps", type=float, default=1.0,
                    help="dense sampling rate for dedup mode (default: 1 fps)")
    ap.add_argument("--phash-threshold", type=int, default=10,
                    help="pHash distance cutoff in dedup mode (higher = fewer keepers)")
    ap.add_argument("--redo-dedup", action="store_true",
                    help="skip ffmpeg; only re-run pHash filter on the cached dense frames")
    ap.add_argument("--static-image-dir", type=str,
                    help="if given, skip ffmpeg; ingest images from this dir instead")
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap on total frames/images ingested")
    args = ap.parse_args()

    city_root = ROOT / "data" / "cities" / args.city
    out_root = city_root / "frames"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.static_image_dir:
        count = ingest_static_dir(Path(args.static_image_dir), out_root / "static", args.max_images)
        print(f"[extract] total: {count} images from static dir")
        return

    videos_dir = city_root / "videos"
    videos = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov")) + list(videos_dir.glob("*.webm"))
    if not videos:
        raise SystemExit(f"no videos in {videos_dir}; use --static-image-dir as fallback")

    total = 0
    for v in videos:
        if args.dedup:
            total += dedup_scene_change(
                v, out_root / v.stem,
                dense_fps=args.dense_fps,
                phash_threshold=args.phash_threshold,
                redo_only=args.redo_dedup,
            )
        else:
            total += extract_with_ffmpeg(v, out_root / v.stem, args.fps)
        # Cap only applies if explicitly set; in dedup mode the natural
        # output size is content-determined and usually the right answer.
        if args.max_images and total >= args.max_images:
            print(f"  [warn] hit --max-images={args.max_images} cap; "
                  f"trimming {total - args.max_images} trailing frames")
            for k in sorted((out_root / v.stem).glob("*.jpg"))[args.max_images:]:
                k.unlink()
            total = args.max_images
            break
    print(f"[extract] total: {total} frames from {len(videos)} videos "
          f"(mode={'dedup' if args.dedup else 'fixed'})")


if __name__ == "__main__":
    main()
