"""End-to-end pipeline for newly-extracted video frames.

After `extract_extra_videos.py` produces new dirs under
`data/cities/<city>/frames/extra_*`, this script runs each through:

  1. DINOv2 embedding (embed_images.py, --method avg)
  2. Visual match GPS  (visual_match_gps.py against the city's mapillary refs)
  3. Refine (refine_visual_match.py)  — GPS+compass consensus
  4. OCR + landmark match (ocr_augment.py + landmark_match.py)
  5. Heading from compass median (compute_frame_heading.py)
  6. Append confidence_v2=high frames to a combined trusted_starts file

Each video subdir gets its own intermediate files; final outputs are merged.

Usage
-----
    python toolbox/process_extra_videos.py \\
        --city zurich \\
        --reference data/cities/mapillary/zurich/embeddings.npz \\
        --mly-meta  data/cities/mapillary/zurich/meta.jsonl \\
        --vllm-url  http://localhost:8003/v1
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, **kw):
    """Run a subprocess, abort on non-zero unless allow_fail=True."""
    allow_fail = kw.pop("allow_fail", False)
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(cmd, **kw)
    if res.returncode != 0 and not allow_fail:
        raise SystemExit(f"step failed: {' '.join(str(c) for c in cmd)}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="zurich")
    ap.add_argument("--frames-root", default=None,
                    help="parent of extra_* dirs; default data/cities/<city>/frames")
    ap.add_argument("--reference",
                    default="data/cities/mapillary/zurich/embeddings.npz",
                    help="Mapillary reference embeddings .npz")
    ap.add_argument("--mly-meta",
                    default="data/cities/mapillary/zurich/meta.jsonl",
                    help="Mapillary meta.jsonl (for compass lookups)")
    ap.add_argument("--vllm-url", default="http://localhost:8003/v1")
    ap.add_argument("--ocr-backend", choices=["paddle", "gemma", "skip"],
                    default="paddle",
                    help="paddle = fast PaddleOCR (~0.2s/frame), "
                         "gemma = old slow vLLM Gemma OCR (~5s/frame), "
                         "skip = no OCR leg")
    ap.add_argument("--ocr-model", default="google/gemma-4-31b-it",
                    help="only used when --ocr-backend=gemma")
    ap.add_argument("--device", default="cuda:0")
    # backwards compat
    ap.add_argument("--skip-ocr", action="store_true",
                    help="alias for --ocr-backend=skip")
    ap.add_argument("--out-trusted",
                    default="data/cities/zurich/frame_starts_trusted_extra.jsonl",
                    help="combined trusted_starts file for all extra videos")
    args = ap.parse_args()

    root = Path(args.frames_root or f"data/cities/{args.city}/frames")
    # Exclude _dense (ffmpeg intermediate cache; we embed only the deduped set)
    extra_dirs = sorted([p for p in root.glob("extra_*") if p.is_dir()
                         and not p.name.endswith("_dense")
                         and any(p.glob("*.jpg"))])
    if not extra_dirs:
        raise SystemExit(f"no extra_* dirs under {root}")
    print(f"[extra] {len(extra_dirs)} extra video dirs to process")
    for d in extra_dirs:
        print(f"  {d.name} ({len(list(d.glob('*.jpg')))} frames)")

    combined = []   # rows that will be written to out-trusted
    for d in extra_dirs:
        name = d.name  # e.g. "extra_Zurich_4K_..."
        print(f"\n========== {name} ==========")
        emb_npz = d.parent / f"{name}_embeddings.npz"
        gps_jsonl = d.parent.parent / f"{name}_frame_gps.jsonl"
        gps_refined = d.parent.parent / f"{name}_frame_gps_refined.jsonl"
        gps_ocr_txt = d.parent.parent / f"{name}_frame_ocr.jsonl"
        gps_ocr_match = d.parent.parent / f"{name}_frame_gps_ocr.jsonl"
        gps_heading = d.parent.parent / f"{name}_frame_heading.jsonl"

        # 1. embed
        if not emb_npz.exists():
            run([sys.executable, "toolbox/embed_images.py", "--method", "avg",
                 "--images", str(d), "--out", str(emb_npz),
                 "--device", args.device])
        else:
            print(f"  [skip embed] {emb_npz.name} exists")

        # 2. visual match
        if not gps_jsonl.exists():
            run([sys.executable, "toolbox/visual_match_gps.py",
                 "--reference", args.reference,
                 "--query", str(emb_npz),
                 "--out", str(gps_jsonl),
                 "--topk", "5"])
        else:
            print(f"  [skip vmatch] {gps_jsonl.name} exists")

        # 3. refine
        if not gps_refined.exists():
            run([sys.executable, "toolbox/refine_visual_match.py",
                 "--frame-gps", str(gps_jsonl),
                 "--mly-meta", args.mly_meta,
                 "--out", str(gps_refined)])
        else:
            print(f"  [skip refine] {gps_refined.name} exists")

        # 4. heading (from top-K compass median)
        if not gps_heading.exists():
            run([sys.executable, "toolbox/compute_frame_heading.py",
                 "--frame-gps", str(gps_jsonl),
                 "--mly-meta", args.mly_meta,
                 "--out", str(gps_heading)])
        else:
            print(f"  [skip heading] {gps_heading.name} exists")

        # 5. OCR (paddle is fast, gemma is slow, skip skips)
        ocr_backend = "skip" if args.skip_ocr else args.ocr_backend
        if ocr_backend != "skip":
            if not gps_ocr_txt.exists():
                if ocr_backend == "paddle":
                    run([sys.executable, "toolbox/ocr_paddle.py",
                         "--frames", str(d),
                         "--out", str(gps_ocr_txt)])
                else:  # gemma
                    run([sys.executable, "toolbox/ocr_augment.py",
                         "--frames", str(d),
                         "--out", str(gps_ocr_txt),
                         "--vllm-url", args.vllm_url,
                         "--model", args.ocr_model])
            else:
                print(f"  [skip ocr] {gps_ocr_txt.name} exists")
            if not gps_ocr_match.exists():
                run([sys.executable, "toolbox/landmark_match.py",
                     "--ocr", str(gps_ocr_txt),
                     "--out", str(gps_ocr_match)])
            else:
                print(f"  [skip lmatch] {gps_ocr_match.name} exists")

        # 6. accumulate trusted starts (delegated to build_trusted_starts.py
        # called per-video; we just collect later by appending all videos
        # into one combined run)
        combined.append({
            "video": name,
            "ocr": str(gps_ocr_match) if ocr_backend != "skip" else None,
            "refined": str(gps_refined),
            "heading": str(gps_heading),
            "frames_dir": str(d),
        })

    # 7. one final trusted_starts merge per video, then concatenate
    print("\n========== merge trusted starts ==========")
    final_lines = []
    for entry in combined:
        per_video_out = Path(entry["frames_dir"]).parent.parent / \
                        f"{Path(entry['frames_dir']).name}_trusted.jsonl"
        cmd = [sys.executable, "toolbox/build_trusted_starts.py",
               "--frames-dir", entry["frames_dir"],
               "--refined", entry["refined"],
               "--heading", entry["heading"],
               "--out", str(per_video_out)]
        if entry["ocr"]:
            cmd += ["--ocr", entry["ocr"]]
        else:
            # build_trusted_starts.py loads "" gracefully if file absent
            cmd += ["--ocr", "/dev/null"]
        run(cmd, allow_fail=True)
        if per_video_out.exists():
            final_lines.extend(per_video_out.read_text().splitlines())

    out_trusted = Path(args.out_trusted)
    out_trusted.parent.mkdir(parents=True, exist_ok=True)
    out_trusted.write_text("\n".join(l for l in final_lines if l.strip()) + "\n")
    print(f"\n[extra] {len(final_lines)} trusted starts → {out_trusted}")
    print(f"\n   Next step:  cat {out_trusted} >> "
          "data/cities/zurich/frame_starts_trusted.jsonl")
    print(f"               then re-run synth_unified.py")


if __name__ == "__main__":
    main()
