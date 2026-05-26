"""Heading QC — drop frames whose recovered heading is untrustworthy.

This is the **last filter** in Phase A. Inputs:

  data/cities/zurich/gps_recovery_all.jsonl   (per-frame GPS+heading)
  data/cities/zurich/phaseA_snapped.jsonl     (optional — adds the
                                               HMM-snapped position and
                                               the segment_bearing)

A frame survives only when **both** are true:

  Q1.  heading_gap >= HEADING_GAP_MIN        (the per-frame DINOv2
                                              top-1 vs 2nd-best margin
                                              at top-1's pano is large
                                              enough that "which way
                                              do I face?" is decidable)

  Q2.  if a segment_bearing is present:
          |angle_diff(heading, segment_bearing)|  <=  HEADING_VS_SEG_MAX
       (HMM agrees the camera faces along the snapped walking edge;
       large disagreement = HMM put us on a parallel street or the
       per-frame heading was a 90/180 flip — drop it.)

When HMM has not been run yet, only Q1 is enforced — useful for an
early dataset preview before the full snap+QC chain exists.

Output: data/cities/zurich/phaseA_trusted.jsonl, one row per kept
frame:

  {video, frame_id, gps:[lat,lon], heading, heading_gap, tier,
   segment_bearing?, segment_id?, source_row_idx}

`source_row_idx` is the 0-based line index in gps_recovery_all.jsonl
so the trusted row can be joined back to the full per-frame log.

Pure helpers (gap_pass, bearing_pass) are unit-tested; main reads
files.

  python -m src.heading_qc                      # default thresholds
  python -m src.heading_qc --min-gap 0.05 --max-bearing-deg 60
  python -m src.heading_qc --no-hmm             # run before HMM exists
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402
from src.routing import angle_diff             # noqa: E402

HEADING_GAP_MIN = 0.05         # below = front/back symmetric, heading ambiguous
HEADING_VS_SEG_MAX = 60.0      # deg — HMM agreement tolerance


def gap_pass(heading_gap, min_gap=HEADING_GAP_MIN) -> bool:
    """Q1 — same-pano top-1 vs 2nd-best margin is large enough."""
    return heading_gap is not None and heading_gap >= min_gap


def bearing_pass(heading, segment_bearing,
                 max_deg=HEADING_VS_SEG_MAX) -> bool:
    """Q2 — camera heading roughly along the HMM-snapped edge."""
    if segment_bearing is None:
        return True            # HMM contributed nothing — let Q1 decide
    return abs(angle_diff(heading, segment_bearing)) <= max_deg


def _load_snapped(snapped_path):
    """{ (video, frame_id) -> {gps, segment_bearing, segment_id} }.
    Returns {} if the file does not exist.

    Accepts either the new road_snap.py schema (`gps_snapped`,
    `segment_bearing`, `segment_id`) or any older variant that still
    carries those keys directly."""
    out = {}
    if not snapped_path.exists():
        return out
    for line in snapped_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        # accept both key names
        gps = d.get("gps_snapped") or d.get("gps")
        out[(d["video"], d["frame_id"])] = {
            "gps": gps,
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
                    help="HMM road-snap output (optional — Q1-only "
                         "when absent)")
    ap.add_argument("--output",
                    default=str(config.CITY_DIR / "trusted_frames.jsonl"))
    ap.add_argument("--min-gap", type=float, default=HEADING_GAP_MIN)
    ap.add_argument("--max-bearing-deg", type=float,
                    default=HEADING_VS_SEG_MAX)
    ap.add_argument("--no-hmm", action="store_true",
                    help="Ignore the snapped file even if present.")
    args = ap.parse_args()

    in_path = Path(args.input)
    snap_path = Path(args.snapped)
    out_path = Path(args.output)

    snapped = {} if args.no_hmm else _load_snapped(snap_path)
    print(f"[heading_qc] in:      {in_path}", flush=True)
    print(f"[heading_qc] snapped: {snap_path}  "
          f"({'used' if snapped else 'missing — Q1 only'})", flush=True)

    n_in = n_accepted = 0
    n_kept = 0
    drop_gap = drop_bearing = 0
    by_video = {}
    with in_path.open(encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            n_in += 1
            r = json.loads(line)
            if not r.get("accepted"):
                continue
            n_accepted += 1
            heading = r.get("heading", 0.0)
            heading_gap = r.get("heading_gap")
            snap = snapped.get((r["video"], r["frame_id"]), {})
            seg_b = snap.get("segment_bearing")

            if not gap_pass(heading_gap, args.min_gap):
                drop_gap += 1
                continue
            if not bearing_pass(heading, seg_b, args.max_bearing_deg):
                drop_bearing += 1
                continue

            out = {
                "video": r["video"],
                "frame_id": r["frame_id"],
                "gps": snap.get("gps") or r["gps"],   # prefer snapped
                "heading": heading,
                "heading_gap": heading_gap,
                "tier": r.get("tier"),
                "source_row_idx": idx,
            }
            if seg_b is not None:
                out["segment_bearing"] = seg_b
                out["segment_id"] = snap.get("segment_id")
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_kept += 1
            by_video[r["video"]] = by_video.get(r["video"], 0) + 1

    print(f"[heading_qc] rows in:        {n_in}", flush=True)
    print(f"[heading_qc] accepted (Phase A): {n_accepted}", flush=True)
    print(f"[heading_qc] dropped — Q1 gap<{args.min_gap}: {drop_gap}",
          flush=True)
    print(f"[heading_qc] dropped — Q2 |Δbearing|>{args.max_bearing_deg}°: "
          f"{drop_bearing}", flush=True)
    print(f"[heading_qc] kept (phaseA_trusted): {n_kept}", flush=True)
    print(f"[heading_qc] per-video: " +
          ", ".join(f"{v}={c}" for v, c in sorted(by_video.items())),
          flush=True)
    print(f"[heading_qc] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
