"""Expand the training frame set without losing existing annotations.

Given:
  - kept_dir : frames already annotated (zurich/)
  - eval_dir : held-out frames we must NOT leak (zurich_eval/)
  - dense_dir: raw 1-fps cache (zurich_dense/)

Walk the dense frames in order; for each:
  a) Compute pHash.
  b) Skip if near-duplicate (distance < near_thresh) to any pHash already in
     kept_dir, eval_dir, OR any new frame we just added.
  c) If pHash distance from the LAST ADDED frame is >= add_thresh, add it.

Output: frames are copied into kept_dir with fresh filenames
`frame_xNNNNN.jpg` so they don't collide with the existing `frame_00000.jpg`
series.
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image
import imagehash

ROOT = Path("/pub/evaluation_group/ning/test/navlm")


def phashes(d: Path):
    out = []
    for f in sorted(d.glob("*.jpg")):
        out.append((f, imagehash.phash(Image.open(f))))
    return out


def nearest_dist(h, hashes):
    return min((h - ph[1] for ph in hashes), default=999)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", required=True, help="dir of already-annotated frames")
    ap.add_argument("--eval", required=True, help="dir of held-out eval frames (to avoid leaking)")
    ap.add_argument("--dense", required=True, help="dir of 1-fps dense cache")
    ap.add_argument("--near-thresh", type=int, default=5,
                    help="phash distance below which a dense frame counts as already covered")
    ap.add_argument("--add-thresh", type=int, default=10,
                    help="phash distance required between two newly added frames")
    args = ap.parse_args()

    kept = Path(args.kept)
    evald = Path(args.eval)
    dense = Path(args.dense)

    print(f"[hashing] existing kept frames in {kept}")
    kept_hashes = phashes(kept)
    print(f"  {len(kept_hashes)}")
    print(f"[hashing] eval frames in {evald}")
    eval_hashes = phashes(evald)
    print(f"  {len(eval_hashes)}")
    print(f"[hashing] dense frames in {dense}")
    dense_files = sorted(dense.glob("*.jpg"))
    print(f"  {len(dense_files)}")

    # Find next available filename index
    existing_x_idx = [
        int(p.stem.split("_x")[1])
        for p in kept.glob("frame_x*.jpg")
        if p.stem.split("_x")[1].isdigit()
    ]
    start_idx = max(existing_x_idx, default=-1) + 1

    added_hashes = []  # list of pHash of newly added frames (for intra-dedup)
    last_added = None
    added_count = 0
    skipped_dup_kept = skipped_dup_eval = skipped_dup_new = skipped_too_similar = 0

    for f in dense_files:
        h = imagehash.phash(Image.open(f))
        # Check overlap with kept
        if min((h - ph for _, ph in kept_hashes), default=999) < args.near_thresh:
            skipped_dup_kept += 1
            continue
        # Check overlap with eval
        if min((h - ph for _, ph in eval_hashes), default=999) < args.near_thresh:
            skipped_dup_eval += 1
            continue
        # Check overlap with already-added new frames
        if min((h - ah for ah in added_hashes), default=999) < args.near_thresh:
            skipped_dup_new += 1
            continue
        # Enforce scene change from last added (keep delta diverse)
        if last_added is not None and (h - last_added) < args.add_thresh:
            skipped_too_similar += 1
            continue
        target = kept / f"frame_x{start_idx + added_count:05d}.jpg"
        shutil.copy2(f, target)
        added_hashes.append(h)
        last_added = h
        added_count += 1

    print(f"\n[expand] added {added_count} new frames to {kept}")
    print(f"  skipped (near-dup of already-kept):   {skipped_dup_kept}")
    print(f"  skipped (near-dup of eval):           {skipped_dup_eval}")
    print(f"  skipped (near-dup of newly added):    {skipped_dup_new}")
    print(f"  skipped (too similar to last added):  {skipped_too_similar}")
    print(f"  total frames in {kept.name} now:      {len(list(kept.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
