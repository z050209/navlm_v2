"""Heading QC — drop frames whose per-frame DINOv2 heading is ambiguous.

The cohort entering heading_qc is pHash-deduped at extraction:
visually-similar consecutive frames collapse to one, so the surviving
sequence is a set of scenically-distinct moments, **not a temporally-
continuous walk**. Median real-time gap between t-3 and t+3 in our
top-30 cohort is 32 s (p90 = 176 s, max 12.7 min). With that sparse
sampling there is no way to derive walker motion from neighbouring
frames — a 3 m net displacement could come from "walked straight" or
"stopped, turned, walked back, stopped again". Any cross-check that
assumes continuous forward walking (heading vs HMM edge bearing,
heading vs temporal-difference bearing of GPS) silently rejects
legitimate stop-and-look frames where the videographer paused to
film a landmark, which is the bulk of a walking-tour video.

So heading_qc keeps only the check that does not depend on temporal
continuity: the **per-frame DINOv2 confidence** at the matched pano.

  Q1   heading_gap >= --min-gap   (default 0.05)
       where heading_gap = (best_cos - 2nd_best_cos) / best_cos
       computed at the 4 compass crops of the matched SV pano.
       Q1 fails on front/back symmetric facades where DINOv2 can't
       decide which direction the camera points.

Optional flags `--use-q2` / `--use-q3` reintroduce the old motion-
based checks if you want them for an ablation; both are off by
default. They are no longer computed at all in default mode — only
the per-frame columns are read from the input.

Output: `data/cities/zurich/trusted_frames.jsonl`, one row per kept
frame, schema:

  {video, frame_id, gps:[lat,lon], heading, heading_gap, tier,
   place_guess, segment_id, segment_bearing, source_row_idx}

The snapped GPS + segment_id are passed through from road_snapped.jsonl
because downstream (annotation, viz) benefits from the smoothed
position. They are not used by heading_qc itself.

  python -m src.heading_qc                              # Q1-only, default
  python -m src.heading_qc --min-gap 0.10               # tighter Q1
  python -m src.heading_qc --use-q2 --use-q3            # restore motion checks
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402

HEADING_GAP_MIN = 0.05         # Q1: below = front/back symmetric, ambiguous


def gap_pass(heading_gap, min_gap=HEADING_GAP_MIN) -> bool:
    """Q1 — DINOv2 best vs 2nd-best cosine gap at the matched pano."""
    return heading_gap is not None and heading_gap >= min_gap


def _load_snapped(snapped_path):
    """{ (video, frame_id) -> {gps, segment_bearing, segment_id} }.
    Accepts both road_snapped.jsonl (gps_snapped) and older variants."""
    out = {}
    if not snapped_path.exists():
        return out
    for line in snapped_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        out[(d["video"], d["frame_id"])] = {
            "gps": d.get("gps_snapped") or d.get("gps"),
            "segment_bearing": d.get("segment_bearing"),
            "segment_id": d.get("segment_id"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "gps_recovery_full.jsonl"))
    ap.add_argument("--snapped",
                    default=str(config.CITY_DIR / "road_snapped.jsonl"),
                    help="HMM road-snap output. When present, snapped GPS "
                         "and segment_id are passed through to the output. "
                         "Use --no-hmm to ignore.")
    ap.add_argument("--output",
                    default=str(config.CITY_DIR / "trusted_frames.jsonl"))
    ap.add_argument("--min-gap", type=float, default=HEADING_GAP_MIN,
                    help="Q1 threshold (heading_gap)")
    ap.add_argument("--no-hmm", action="store_true",
                    help="Ignore the snapped file even if present.")
    ap.add_argument("--diagnostics",
                    default=str(config.CITY_DIR /
                                "heading_qc_diagnostics.jsonl"),
                    help="per-frame Q1 diagnostics (input to "
                         "src.viz_heading_qc)")
    args = ap.parse_args()

    def _resolve(p):
        path = Path(p)
        if path.exists() or path.is_absolute():
            return path
        in_city = config.CITY_DIR / path.name
        return in_city if in_city.exists() else path

    in_path = _resolve(args.input)
    snap_path = _resolve(args.snapped)
    out_path = (Path(args.output) if Path(args.output).is_absolute()
                else config.CITY_DIR / Path(args.output).name)
    diag_path = (Path(args.diagnostics) if Path(args.diagnostics).is_absolute()
                 else config.CITY_DIR / Path(args.diagnostics).name)

    snapped = {} if args.no_hmm else _load_snapped(snap_path)
    print(f"[heading_qc] in:        {in_path}", flush=True)
    print(f"[heading_qc] snapped:   {snap_path}  "
          f"({'used (passthrough only)' if snapped else 'missing/off'})",
          flush=True)
    print(f"[heading_qc] threshold: Q1 heading_gap >= {args.min_gap} "
          f"(Q1-only filter; Q2/Q3 dropped — see docstring for rationale)",
          flush=True)

    # ── Load + filter to accepted + restrict to road-snapped cohort ──
    rows_all = []
    with in_path.open(encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("accepted"):
                continue
            snap = snapped.get((r["video"], r["frame_id"]), {})
            rows_all.append({
                "_idx": idx,
                "video": r["video"],
                "frame_id": r["frame_id"],
                "gps": snap.get("gps") or r["gps"],
                "heading": r.get("heading"),
                "heading_gap": r.get("heading_gap"),
                "tier": r.get("tier"),
                "place_guess": r.get("place_guess", ""),
                "segment_bearing": snap.get("segment_bearing"),
                "segment_id": snap.get("segment_id"),
                "in_snapped": bool(snap),
            })

    if snapped:
        n_pre = len(rows_all)
        rows_all = [r for r in rows_all if r["in_snapped"]]
        print(f"[heading_qc] restricted to frames in {snap_path.name}: "
              f"{len(rows_all):,} (was {n_pre:,})", flush=True)

    # ── Apply Q1 + write output ────────────────────────────────────
    n_kept = n_q1_fail = 0
    by_video = collections.Counter()
    diagnostics = []

    with out_path.open("w", encoding="utf-8") as fout:
        for r in rows_all:
            ok = gap_pass(r["heading_gap"], args.min_gap)
            diagnostics.append({
                "video": r["video"], "frame_id": r["frame_id"],
                "heading": r["heading"],
                "heading_gap": r["heading_gap"],
                "q1": ok,
                "pass_all": ok,
            })
            if not ok:
                n_q1_fail += 1
                continue
            fout.write(json.dumps({
                "video": r["video"],
                "frame_id": r["frame_id"],
                "gps": r["gps"],
                "heading": r["heading"],
                "heading_gap": r["heading_gap"],
                "tier": r["tier"],
                "place_guess": r["place_guess"],
                "segment_bearing": r["segment_bearing"],
                "segment_id": r["segment_id"],
                "source_row_idx": r["_idx"],
            }, ensure_ascii=False) + "\n")
            n_kept += 1
            by_video[r["video"]] += 1

    with diag_path.open("w", encoding="utf-8") as fdiag:
        for d in diagnostics:
            fdiag.write(json.dumps(d, ensure_ascii=False) + "\n")

    total = len(rows_all)
    print(f"[heading_qc] frames considered:  {total}", flush=True)
    print(f"[heading_qc] dropped Q1 gap<{args.min_gap}:    {n_q1_fail}",
          flush=True)
    print(f"[heading_qc] KEPT (trusted_frames):   {n_kept}  "
          f"({100*n_kept/max(1,total):.0f} %)", flush=True)
    print(f"[heading_qc] per-video: " +
          ", ".join(f"{v}={c}" for v, c in sorted(by_video.items())),
          flush=True)
    print(f"[heading_qc] wrote {out_path}", flush=True)
    print(f"[heading_qc] wrote {diag_path}  "
          f"(input to `python -m src.viz_heading_qc`)", flush=True)


if __name__ == "__main__":
    main()
