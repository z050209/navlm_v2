"""POI + heading distribution charts for the VLM-agreed cohort.

For sanity-checking the 2,470 VLM-agreed accepted frames produced by
`gps_recovery --poi-scan poi_scan_cos0.75.jsonl`:

  1. POI bar chart   — which OSM POIs (via DINOv2-nearest) the 2,470
                       frames are anchored to, sorted by count. Tells
                       us whether the cohort is concentrated on a few
                       landmarks or spread across the city.
  2. Heading rose    — circular histogram of the per-frame recovered
                       heading (degrees, 0 = N). Tells us whether the
                       camera headings are well-distributed or skewed
                       to a few directions (which would hint at video
                       framing bias).
  3. Heading linear  — plain histogram of headings in 10° bins, for
                       absolute counts at each angle.

  python -m src.viz_distributions
  python -m src.viz_distributions --input gps_recovery_full.jsonl --tier 1
  python -m src.viz_distributions --top-n 50         # show top 50 POIs
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402


def load_rows(path, tier, accepted_only):
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if accepted_only and not r.get("accepted"):
            continue
        if tier and r.get("tier") != tier:
            continue
        rows.append(r)
    return rows


def plot_poi_bar(rows, out_path, top_n=30):
    """Horizontal bar chart of POI -> frame-count. Uses
    `dino_nearest_name` as the POI label (the OSM POI nearest the
    matched Street View pano)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = collections.Counter(r.get("dino_nearest_name", "") or "—"
                                  for r in rows)
    common = counts.most_common(top_n)
    names = [n for n, _ in common][::-1]            # bottom -> top
    vals = [c for _, c in common][::-1]
    total = sum(counts.values())
    shown = sum(vals)
    distinct = len(counts)

    h = max(4, 0.28 * len(names) + 1.0)
    fig, ax = plt.subplots(figsize=(8, h))
    bars = ax.barh(names, vals, color="#2a9d8f", edgecolor="#1d6a5d")
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.005, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=8)
    ax.set_xlabel(f"frame count (top {top_n} of {distinct} POIs; "
                  f"shown {shown}/{total} = {100*shown/total:.0f}%)")
    ax.set_title(f"POI distribution — {len(rows)} VLM-agreed frames\n"
                 f"(POI = DINOv2-nearest OSM POI to the matched SV pano)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle=":", color="#aaa", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return distinct, total


def plot_heading_rose(rows, out_path, bin_deg=15):
    """Circular polar histogram of the per-frame heading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    headings = [r.get("heading") for r in rows
                if r.get("heading") is not None]
    n_bins = 360 // bin_deg
    bins = np.linspace(0, 360, n_bins + 1)
    counts, _ = np.histogram(headings, bins=bins)
    theta = np.deg2rad(bins[:-1] + bin_deg / 2.0)
    width = np.deg2rad(bin_deg)

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)              # clockwise (compass-style)
    bars = ax.bar(theta, counts, width=width, bottom=0.0,
                  color="#3a86ff", edgecolor="#22487f", alpha=0.85)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_title(f"Heading distribution — {len(headings)} VLM-agreed frames\n"
                 f"(per-frame recovered heading, {bin_deg}° bins; 0°=N, clockwise)",
                 pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return counts.max(), counts.sum()


def plot_heading_linear(rows, out_path, bin_deg=10):
    """Plain heading histogram (0-360°) — easier to read exact counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    headings = [r.get("heading") for r in rows
                if r.get("heading") is not None]
    bins = np.arange(0, 361, bin_deg)
    counts, edges = np.histogram(headings, bins=bins)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(edges[:-1], counts, width=bin_deg, align="edge",
           color="#3a86ff", edgecolor="#22487f")
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel("heading (degrees, 0=N, clockwise)")
    ax.set_ylabel("frame count")
    ax.set_title(f"Heading distribution — {len(headings)} VLM-agreed frames "
                 f"({bin_deg}° bins)")
    ax.axvspan(45 - 22.5, 45 + 22.5, alpha=0.06, color="orange")
    ax.axvspan(135 - 22.5, 135 + 22.5, alpha=0.06, color="orange")
    ax.axvspan(225 - 22.5, 225 + 22.5, alpha=0.06, color="orange")
    ax.axvspan(315 - 22.5, 315 + 22.5, alpha=0.06, color="orange")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", color="#aaa", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return counts.max(), counts.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input", default="gps_recovery_full.jsonl")
    ap.add_argument("--tier", type=int, choices=[0, 1, 2], default=1,
                    help="filter rows by tier (default 1 = VLM-agreed only)")
    ap.add_argument("--top-n", type=int, default=30,
                    help="POIs to show in the bar chart")
    ap.add_argument("--bin-deg", type=int, default=15,
                    help="heading bin width in degrees (rose plot)")
    ap.add_argument("--prefix", default="vlm_agreed",
                    help="output filename prefix under viz/")
    args = ap.parse_args()

    in_path = config.CITY_DIR / args.input
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")
    rows = load_rows(in_path, args.tier, accepted_only=True)
    print(f"[viz_distributions] loaded {len(rows)} rows from "
          f"{in_path.name} (tier={args.tier}, accepted=True)")

    viz_dir = config.VIZ_DIR
    viz_dir.mkdir(parents=True, exist_ok=True)

    poi_path = viz_dir / f"poi_distribution_{args.prefix}.png"
    distinct, total = plot_poi_bar(rows, poi_path, top_n=args.top_n)
    print(f"[viz_distributions] {distinct} distinct POIs across "
          f"{total} sightings -> {poi_path}")

    rose_path = viz_dir / f"heading_rose_{args.prefix}.png"
    rmax, rsum = plot_heading_rose(rows, rose_path, bin_deg=args.bin_deg)
    print(f"[viz_distributions] heading rose: peak {rmax} in one "
          f"{args.bin_deg}° bin, {rsum} frames -> {rose_path}")

    lin_path = viz_dir / f"heading_linear_{args.prefix}.png"
    lmax, lsum = plot_heading_linear(rows, lin_path, bin_deg=10)
    print(f"[viz_distributions] heading linear: peak {lmax} in one "
          f"10° bin -> {lin_path}")


if __name__ == "__main__":
    main()
