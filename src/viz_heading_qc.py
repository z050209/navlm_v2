"""Diagnostic plots for heading_qc — explains the three-bearing check.

Reads `data/cities/zurich/heading_qc_diagnostics.jsonl` (written by
`src.heading_qc`) and writes a set of PNGs to `viz/`:

  viz/heading_qc_dropreasons.png       — bar chart: kept vs each Q drop
  viz/heading_qc_delta_seg_hist.png    — |recovered − segment| histogram
  viz/heading_qc_delta_td_hist.png     — |recovered − td|       histogram
  viz/heading_qc_delta_joint.png       — 2D scatter of (Δseg, Δtd)
                                          colour = pass_all
  viz/heading_qc_pervideo.png          — pass rate per video

  python -m src.viz_heading_qc
  python -m src.viz_heading_qc --input heading_qc_diagnostics.jsonl
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

    # ─── (1) drop-reasons bar ──────────────────────────────────────
    # First-fail counting: drop in order Q1 -> Q2 -> Q3 (matches
    # heading_qc.py logic so the bars add up to total).
    kept = 0; dropped_q1 = 0; dropped_q2 = 0; dropped_q3 = 0
    for r in rows:
        if not r["q1"]:
            dropped_q1 += 1
        elif not r["q2"]:
            dropped_q2 += 1
        elif not r["q3"]:
            dropped_q3 += 1
        else:
            kept += 1
    total = max(1, len(rows))
    labels = [f"KEPT\n(all 3 pass)\n{kept}\n{100*kept/total:.0f} %",
              f"Q1 fail\n(heading_gap<0.05)\n{dropped_q1}\n"
              f"{100*dropped_q1/total:.0f} %",
              f"Q2 fail\n(|Δseg|>60°)\n{dropped_q2}\n"
              f"{100*dropped_q2/total:.0f} %",
              f"Q3 fail\n(|Δtd|>60°)\n{dropped_q3}\n"
              f"{100*dropped_q3/total:.0f} %"]
    vals = [kept, dropped_q1, dropped_q2, dropped_q3]
    colors = ["#2a9d8f", "#cccccc", "#f4a261", "#e76f51"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(4), vals, color=colors, edgecolor="#333")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("frames")
    ax.set_title(f"heading_qc — {len(rows)} VLM-agreed + top-30 POI frames\n"
                 f"first-fail counting (Q1 → Q2 → Q3)")
    ax.set_axisbelow(True); ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = viz_dir / f"{args.prefix}_dropreasons.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  -> {out}")

    # ─── (2) |Δseg| histogram (Q2) ─────────────────────────────────
    deltas_seg = [abs(r["delta_seg"]) for r in rows
                  if r["delta_seg"] is not None]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(deltas_seg, bins=np.arange(0, 185, 5),
            color="#f4a261", edgecolor="#7a4a26")
    ax.axvline(60, color="#c00", ls="--", lw=2,
               label="Q2 threshold (60°)")
    ax.set_xlabel("|heading_recovered − segment_bearing|  (deg)")
    ax.set_ylabel("frames")
    ax.set_title(f"Q2: per-frame heading vs HMM edge bearing "
                 f"(N={len(deltas_seg)})")
    ax.legend(); ax.set_axisbelow(True); ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = viz_dir / f"{args.prefix}_delta_seg_hist.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  -> {out}")

    # ─── (3) |Δtd| histogram (Q3) ──────────────────────────────────
    deltas_td = [abs(r["delta_td"]) for r in rows
                 if r["delta_td"] is not None]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(deltas_td, bins=np.arange(0, 185, 5),
            color="#e76f51", edgecolor="#7a3624")
    ax.axvline(60, color="#c00", ls="--", lw=2,
               label="Q3 threshold (60°)")
    ax.set_xlabel("|heading_recovered − td_bearing|  (deg)")
    ax.set_ylabel("frames")
    ax.set_title(f"Q3: per-frame heading vs walker's actual motion "
                 f"(N={len(deltas_td)})")
    ax.legend(); ax.set_axisbelow(True); ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = viz_dir / f"{args.prefix}_delta_td_hist.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  -> {out}")

    # ─── (4) joint scatter (Δseg vs Δtd) ───────────────────────────
    paired = [(r["delta_seg"], r["delta_td"], r["pass_all"])
              for r in rows
              if r["delta_seg"] is not None and r["delta_td"] is not None]
    if paired:
        xs = [abs(p[0]) for p in paired]
        ys = [abs(p[1]) for p in paired]
        cs = ["#2a9d8f" if p[2] else "#e76f51" for p in paired]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(xs, ys, c=cs, s=8, alpha=0.5)
        ax.axvline(60, color="#888", ls="--", lw=1)
        ax.axhline(60, color="#888", ls="--", lw=1)
        ax.set_xlim(0, 180); ax.set_ylim(0, 180)
        ax.set_xlabel("|Δseg|  recovered vs HMM edge bearing (deg)")
        ax.set_ylabel("|Δtd|   recovered vs walker motion bearing (deg)")
        ax.set_title("Joint Q2/Q3 disagreement\n"
                     f"green = pass_all,  red = at least one fail "
                     f"(N={len(paired)})")
        ax.text(30, 170, "kept", color="#2a9d8f", fontsize=12,
                ha="center")
        ax.text(120, 170, "Q2 reject", color="#7a3624", fontsize=10,
                ha="center")
        ax.text(30, 120, "Q3 reject", color="#7a3624", fontsize=10,
                ha="center")
        ax.text(120, 120, "both reject", color="#7a3624", fontsize=10,
                ha="center")
        ax.set_axisbelow(True); ax.grid(ls=":", alpha=0.4)
        fig.tight_layout()
        out = viz_dir / f"{args.prefix}_delta_joint.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  -> {out}")

    # ─── (5) per-video pass-rate bar ───────────────────────────────
    per_video_total = collections.Counter()
    per_video_pass = collections.Counter()
    for r in rows:
        per_video_total[r["video"]] += 1
        if r["pass_all"]:
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
    ax.set_title("heading_qc — pass rate per video "
                 "(kept / considered)")
    ax.set_axisbelow(True); ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = viz_dir / f"{args.prefix}_pervideo.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  -> {out}")

    # ─── tiny summary ──────────────────────────────────────────────
    print()
    print(f"summary:  N={len(rows)}  kept={kept}  "
          f"Q1_fail={dropped_q1}  Q2_fail={dropped_q2}  "
          f"Q3_fail={dropped_q3}")
    if deltas_seg:
        ds = sorted(deltas_seg)
        print(f"|d_seg|  median={ds[len(ds)//2]:.1f} deg  "
              f"p90={ds[int(0.9*len(ds))]:.1f} deg  "
              f"max={ds[-1]:.1f} deg")
    if deltas_td:
        dt = sorted(deltas_td)
        print(f"|d_td|   median={dt[len(dt)//2]:.1f} deg  "
              f"p90={dt[int(0.9*len(dt))]:.1f} deg  "
              f"max={dt[-1]:.1f} deg")


if __name__ == "__main__":
    main()
