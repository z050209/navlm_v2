"""Diagnostic plots for heading_qc — Q1 only.

Reads `data/cities/zurich/heading_qc_diagnostics.jsonl` (written by
`src.heading_qc`) and writes PNGs to `viz/`:

  viz/heading_qc_dropreasons.png      — bar: KEPT vs Q1 fail
  viz/heading_qc_gap_hist.png         — histogram of heading_gap
                                         with the 0.05 threshold line
  viz/heading_qc_pervideo.png         — pass rate per video

  python -m src.viz_heading_qc
"""

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR /
                                "heading_qc_diagnostics.jsonl"))
    ap.add_argument("--prefix", default="heading_qc",
                    help="output filename prefix under viz/")
    ap.add_argument("--min-gap", type=float, default=0.05,
                    help="Q1 threshold to draw on the histogram")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"diagnostics file not found: {in_path}\n"
                 f"  run `python -m src.heading_qc` first")

    rows = [json.loads(l) for l in in_path.open(encoding="utf-8")
            if l.strip()]
    print(f"[viz_heading_qc] loaded {len(rows)} rows from {in_path.name}",
          flush=True)
    viz_dir = config.VIZ_DIR
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ─── drop-reasons bar ─────────────────────────────────────────
    kept = sum(1 for r in rows if r["q1"])
    failed = len(rows) - kept
    total = max(1, len(rows))
    labels = [f"KEPT (Q1 pass)\n{kept}\n{100*kept/total:.0f} %",
              f"Q1 fail\nheading_gap<{args.min_gap}\n{failed}\n"
              f"{100*failed/total:.0f} %"]
    vals = [kept, failed]
    colors = ["#2a9d8f", "#cccccc"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(2), vals, color=colors, edgecolor="#333")
    ax.set_xticks(range(2)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("frames")
    ax.set_title(f"heading_qc — Q1-only filter on {len(rows)} VLM-agreed "
                 f"+ top-30 POI frames")
    ax.set_axisbelow(True); ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = viz_dir / f"{args.prefix}_dropreasons.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  -> {out}")

    # ─── heading_gap histogram ─────────────────────────────────────
    gaps = [r["heading_gap"] for r in rows
            if r.get("heading_gap") is not None]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(gaps, bins=np.linspace(0, 1, 50),
            color="#2a9d8f", edgecolor="#1d6a5d")
    ax.axvline(args.min_gap, color="#c00", ls="--", lw=2,
               label=f"Q1 threshold ({args.min_gap})")
    ax.set_xlabel("heading_gap = (best_cos - 2nd_best_cos) / best_cos "
                  "at the matched pano's 4 compass crops")
    ax.set_ylabel("frames")
    ax.set_title(f"heading_gap distribution (N={len(gaps)})  "
                 f"— anything left of the red line is dropped")
    ax.legend(); ax.set_axisbelow(True); ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = viz_dir / f"{args.prefix}_gap_hist.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  -> {out}")

    # ─── per-video pass rate ───────────────────────────────────────
    per_video_total = collections.Counter()
    per_video_pass = collections.Counter()
    for r in rows:
        per_video_total[r["video"]] += 1
        if r["q1"]:
            per_video_pass[r["video"]] += 1
    vids = sorted(per_video_total)
    pass_rates = [100 * per_video_pass[v] / max(1, per_video_total[v])
                  for v in vids]
    totals = [per_video_total[v] for v in vids]
    kept_counts = [per_video_pass[v] for v in vids]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(range(len(vids)), pass_rates,
                  color="#2a9d8f", edgecolor="#1d6a5d")
    for i, (b, k, t) in enumerate(zip(bars, kept_counts, totals)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{k}/{t}", ha="center", fontsize=9)
    ax.set_xticks(range(len(vids)))
    ax.set_xticklabels(vids, rotation=30, ha="right")
    ax.set_ylim(0, 105); ax.set_ylabel("pass rate (%)")
    ax.set_title("heading_qc Q1 — pass rate per video (kept / considered)")
    ax.set_axisbelow(True); ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = viz_dir / f"{args.prefix}_pervideo.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  -> {out}")

    print()
    print(f"summary:  N={len(rows)}  KEPT={kept}  Q1_fail={failed}")
    if gaps:
        gs = sorted(gaps)
        print(f"heading_gap  median={gs[len(gs)//2]:.3f}  "
              f"p25={gs[len(gs)//4]:.3f}  p75={gs[3*len(gs)//4]:.3f}")


if __name__ == "__main__":
    main()
