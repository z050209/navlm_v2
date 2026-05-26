"""Heading QC — drop frames whose recovered heading is untrustworthy.

This is the **last filter** before instruction annotation. It runs three
independent angular checks per frame and keeps only the frames that
pass all three.

For each frame `t` in a video's chronological sequence we have THREE
bearings that should agree (within tolerance) if the heading is right:

  1. heading_recovered  — per-frame DINOv2 same-pano cosine-weighted
                           mean of the 4 compass crops at top-1's pano
                           (the original heading from gps_recovery.py).
  2. segment_bearing    — bearing of the OSM walking edge the HMM
                           snapped this frame to (from road_snap.py).
  3. td_bearing         — bearing computed from neighbouring snapped
                           GPS: bearing(gps_snapped[t-k], gps_snapped[t+k]),
                           the direction the walker is actually moving.
                           Falls back to forward/backward differences
                           at the sequence edges.

ASCII picture (one frame at GPS x; arrows are bearings):

         heading_recovered          (where the CAMERA points)
                  ^
                  |
                  x ──►  segment_bearing    (where the EDGE goes)
                  |
                  ▼
            td_bearing                       (where the WALKER moves,
                                              from neighbouring frames)

If the walker faces forward while walking forward along the edge,
all three arrows point the same way. Disagreements isolate different
failure modes:

  Q1 fails (heading_gap < 0.05)
        Per-frame DINOv2 can't tell direction (front/back symmetric
        building). heading_recovered is unreliable on its own.
  Q2 fails (|recovered - segment| > 60°)
        Either the per-frame heading is wrong, OR HMM snapped to a
        parallel street next door (locally plausible but wrong edge).
  Q3 fails (|recovered - td| > 60°)
        Camera is rotated off the walking direction — e.g. the
        videographer turned to look at a landmark while walking past.
        Such frames are technically "correct" per-frame but useless
        for our instruction-tuning task (the model would learn to
        give directions to a side view).

We require Q1 AND Q2 AND Q3 (when applicable). Q3 is skipped when the
walker is essentially stationary at t (|gps[t+k] - gps[t-k]| < a few
metres) because the bearing is ill-defined for a near-zero displacement.

Output: `data/cities/zurich/trusted_frames.jsonl`, one row per kept
frame, schema:

  {video, frame_id, gps:[lat,lon], heading, heading_gap, tier,
   segment_bearing, segment_id, td_bearing, td_displacement_m,
   place_guess, source_row_idx}

  python -m src.heading_qc                           # default thresholds
  python -m src.heading_qc --min-gap 0.05 --max-bearing-deg 60
  python -m src.heading_qc --td-window 3 --max-td-deg 60
  python -m src.heading_qc --no-hmm                  # Q1-only (no road_snap.jsonl)
  python -m src.heading_qc --no-td                   # Q1+Q2 only
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402
from src.routing import angle_diff             # noqa: E402

HEADING_GAP_MIN = 0.05         # Q1: below = front/back symmetric → ambiguous
HEADING_VS_SEG_MAX = 60.0      # Q2: deg — recovered vs HMM-edge tolerance
HEADING_VS_TD_MAX = 60.0       # Q3: deg — recovered vs temporal-difference tol
TD_WINDOW = 3                  # frames forward/backward for the TD bearing
TD_MIN_DISPLACEMENT_M = 3.0    # skip Q3 when neighbours are too close


def gap_pass(heading_gap, min_gap=HEADING_GAP_MIN) -> bool:
    """Q1 — same-pano top-1 vs 2nd-best margin is large enough."""
    return heading_gap is not None and heading_gap >= min_gap


def bearing_pass(heading, segment_bearing,
                 max_deg=HEADING_VS_SEG_MAX) -> bool:
    """Q2 — camera heading roughly along the HMM-snapped edge."""
    if segment_bearing is None or heading is None:
        return True            # HMM contributed nothing — let Q1 decide
    return abs(angle_diff(heading, segment_bearing)) <= max_deg


def td_pass(heading, td_bearing, displacement_m,
            max_deg=HEADING_VS_TD_MAX,
            min_displacement_m=TD_MIN_DISPLACEMENT_M) -> bool:
    """Q3 — camera heading roughly along the walker's actual motion
    (bearing from neighbouring snapped GPS). Skipped (returns True)
    when displacement is too small to define a bearing."""
    if (td_bearing is None or heading is None
            or displacement_m is None
            or displacement_m < min_displacement_m):
        return True
    return abs(angle_diff(heading, td_bearing)) <= max_deg


def _bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _haversine_m(la1, lo1, la2, lo2):
    R = 6_371_000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlam = math.radians(lo2 - lo1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def compute_td_bearings(per_video_frames, window=TD_WINDOW):
    """Per-frame (td_bearing, td_displacement_m) for one chronological
    sequence of frames sharing a video, each carrying `gps` = [lat, lon].

    Uses bearing(gps[t-window], gps[t+window]) at the interior; falls
    back to forward (t -> t+w) at the start and backward (t-w -> t) at
    the end. Returns a list aligned to `per_video_frames`."""
    n = len(per_video_frames)
    out = []
    for t in range(n):
        lo = max(0, t - window)
        hi = min(n - 1, t + window)
        if hi == lo:
            out.append((None, None))
            continue
        a_lat, a_lon = per_video_frames[lo]["gps"]
        b_lat, b_lon = per_video_frames[hi]["gps"]
        disp = _haversine_m(a_lat, a_lon, b_lat, b_lon)
        if disp <= 0.01:
            out.append((None, disp))
            continue
        out.append((_bearing_deg(a_lat, a_lon, b_lat, b_lon), disp))
    return out


def _load_snapped(snapped_path):
    """{ (video, frame_id) -> {gps, segment_bearing, segment_id} }.
    Accepts both road_snap.py output (gps_snapped) and any older variant
    that uses `gps` directly."""
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
                    help="HMM road-snap output (optional — Q1+Q3 only when "
                         "absent or --no-hmm)")
    ap.add_argument("--output",
                    default=str(config.CITY_DIR / "trusted_frames.jsonl"))
    ap.add_argument("--min-gap", type=float, default=HEADING_GAP_MIN,
                    help="Q1 threshold (heading_gap)")
    ap.add_argument("--max-bearing-deg", type=float,
                    default=HEADING_VS_SEG_MAX,
                    help="Q2 threshold (|recovered − segment|)")
    ap.add_argument("--max-td-deg", type=float,
                    default=HEADING_VS_TD_MAX,
                    help="Q3 threshold (|recovered − td|)")
    ap.add_argument("--td-window", type=int, default=TD_WINDOW,
                    help="frames forward/backward for the TD bearing")
    ap.add_argument("--no-hmm", action="store_true",
                    help="Ignore the snapped file even if present (Q2 off).")
    ap.add_argument("--no-td", action="store_true",
                    help="Disable Q3 (temporal-difference) check.")
    ap.add_argument("--diagnostics",
                    default=str(config.CITY_DIR /
                                "heading_qc_diagnostics.jsonl"),
                    help="dump per-frame Q1/Q2/Q3 diagnostics (input to "
                         "src.viz_heading_qc)")
    args = ap.parse_args()

    in_path = Path(args.input)
    snap_path = Path(args.snapped)
    out_path = Path(args.output)
    diag_path = Path(args.diagnostics)

    snapped = {} if args.no_hmm else _load_snapped(snap_path)
    print(f"[heading_qc] in:        {in_path}", flush=True)
    print(f"[heading_qc] snapped:   {snap_path}  "
          f"({'used' if snapped else 'missing/disabled — Q2 off'})",
          flush=True)
    print(f"[heading_qc] thresholds:"
          f" Q1 gap≥{args.min_gap}"
          f"  Q2 |Δseg|≤{args.max_bearing_deg}°"
          f"  Q3 |Δtd|≤{args.max_td_deg}° "
          f"(window={args.td_window}, td={'on' if not args.no_td else 'off'})",
          flush=True)

    # ── Load + filter to accepted, prefer snapped GPS where present ──
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
                "gps_raw": r["gps"],
                "heading": r.get("heading"),
                "heading_gap": r.get("heading_gap"),
                "tier": r.get("tier"),
                "place_guess": r.get("place_guess", ""),
                "segment_bearing": snap.get("segment_bearing"),
                "segment_id": snap.get("segment_id"),
                "in_snapped": bool(snap),
            })

    # If snapped exists, restrict to ONLY frames that survived
    # road_snap.py's top-N POI filter (so heading_qc operates on the
    # same cohort the teacher will annotate).
    if snapped:
        n_pre = len(rows_all)
        rows_all = [r for r in rows_all if r["in_snapped"]]
        print(f"[heading_qc] restricted to frames in {snap_path.name}: "
              f"{len(rows_all):,} (was {n_pre:,})", flush=True)

    # ── Group by video for the TD pass ─────────────────────────────
    per_video = collections.defaultdict(list)
    for r in rows_all:
        per_video[r["video"]].append(r)
    for v in per_video:
        per_video[v].sort(key=lambda r: r["frame_id"])

    # compute td bearings and attach
    for v, seq in per_video.items():
        td = compute_td_bearings(seq, window=args.td_window)
        for r, (b, d) in zip(seq, td):
            r["td_bearing"] = b
            r["td_displacement_m"] = d

    # ── Apply filters + write outputs ──────────────────────────────
    n_kept = 0
    n_q1 = n_q2 = n_q3 = 0
    by_video = collections.Counter()
    diagnostics = []

    with out_path.open("w", encoding="utf-8") as fout:
        for v, seq in per_video.items():
            for r in seq:
                pass1 = gap_pass(r["heading_gap"], args.min_gap)
                pass2 = bearing_pass(r["heading"], r["segment_bearing"],
                                     args.max_bearing_deg)
                pass3 = True if args.no_td else td_pass(
                    r["heading"], r["td_bearing"], r["td_displacement_m"],
                    args.max_td_deg)

                diagnostics.append({
                    "video": v, "frame_id": r["frame_id"],
                    "heading": r["heading"],
                    "heading_gap": r["heading_gap"],
                    "segment_bearing": r["segment_bearing"],
                    "td_bearing": r["td_bearing"],
                    "td_displacement_m": r["td_displacement_m"],
                    "delta_seg": (
                        None if r["segment_bearing"] is None
                              or r["heading"] is None
                        else float(angle_diff(r["heading"],
                                              r["segment_bearing"]))),
                    "delta_td": (
                        None if r["td_bearing"] is None
                              or r["heading"] is None
                        else float(angle_diff(r["heading"],
                                              r["td_bearing"]))),
                    "q1": pass1, "q2": pass2, "q3": pass3,
                    "pass_all": bool(pass1 and pass2 and pass3),
                })

                if not pass1:
                    n_q1 += 1; continue
                if not pass2:
                    n_q2 += 1; continue
                if not pass3:
                    n_q3 += 1; continue

                out = {
                    "video": v, "frame_id": r["frame_id"],
                    "gps": r["gps"], "heading": r["heading"],
                    "heading_gap": r["heading_gap"],
                    "tier": r["tier"],
                    "place_guess": r["place_guess"],
                    "segment_bearing": r["segment_bearing"],
                    "segment_id": r["segment_id"],
                    "td_bearing": r["td_bearing"],
                    "td_displacement_m": r["td_displacement_m"],
                    "source_row_idx": r["_idx"],
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_kept += 1
                by_video[v] += 1

    # diagnostics dump (for the viz script)
    with diag_path.open("w", encoding="utf-8") as fdiag:
        for d in diagnostics:
            fdiag.write(json.dumps(d, ensure_ascii=False) + "\n")

    total = sum(1 for v in per_video.values() for _ in v)
    print(f"[heading_qc] frames considered:  {total}", flush=True)
    print(f"[heading_qc] dropped Q1 gap<{args.min_gap}:        {n_q1}",
          flush=True)
    print(f"[heading_qc] dropped Q2 |Δseg|>{args.max_bearing_deg}°:    {n_q2}",
          flush=True)
    if not args.no_td:
        print(f"[heading_qc] dropped Q3 |Δtd|>{args.max_td_deg}°:     {n_q3}",
              flush=True)
    print(f"[heading_qc] KEPT (trusted_frames):  {n_kept}", flush=True)
    print(f"[heading_qc] per-video: " +
          ", ".join(f"{v}={c}" for v, c in sorted(by_video.items())),
          flush=True)
    print(f"[heading_qc] wrote {out_path}", flush=True)
    print(f"[heading_qc] wrote {diag_path}  "
          f"(input to `python -m src.viz_heading_qc`)", flush=True)


if __name__ == "__main__":
    main()
