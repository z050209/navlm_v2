"""Generate 3 paper-ready figures for the NavLM v2 ablation results.

Inputs: eval_pull/ablation_<run_id>/<condition>/per_sample_scored.jsonl
Outputs: docs/figures/fig{1,2,3}_*.png  (300 DPI, paper-ready)

Figures:
  1. fig1_rank_saturation.png   — rank (4/8/16) vs PASS, 3 variants × 2 epochs
  2. fig2_zs_vs_trained.png     — zero-shot vs best-trained PASS bars, per variant
  3. fig3_heading_scatter.png   — derived: predicted heading vs GT heading, with
                                  identity line, error bands, mean |err| annotation

  python -m src.a2_figures --run-dir eval_pull/ablation_20260602_054707
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HEAD_RE = re.compile(r"facing\s+(\d{1,3}(?:\.\d+)?)\s*°", re.I)


def load_summary(run_dir: Path) -> dict:
    """Returns {condition_dir_name: {n, fmt, dir, PASS, h_inf, h_n, rows}}."""
    out = {}
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        p = d / "per_sample_scored.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        if not rows:
            continue
        n = len(rows)
        out[d.name] = {
            "n": n,
            "PASS": sum(1 for r in rows if r.get("PASS")) / n,
            "dir":  sum(1 for r in rows if r.get("direction_pass")) / n,
            "fmt":  sum(1 for r in rows if r.get("format_pass")) / n,
            "rows": rows,
        }
    return out


def parse_condition(name: str):
    """Parse 'trained-heading-derived_r16_e5' → ('derived', 16, 5, 'trained')
    or 'zs-heading-given' → ('given', None, None, 'zs')."""
    if name.startswith("zs-heading-"):
        return name.replace("zs-heading-", ""), None, None, "zs"
    m = re.match(r"trained-heading-(\w+)_r(\d+)_e(\d+)", name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), "trained"
    return None, None, None, None


# ─────────────────────── Figure 1: rank-saturation ─────────────────────────
def fig1_rank_saturation(summary, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    variants = ["given", "derived", "implicit"]
    ranks = [4, 8, 16]
    colors = {"given": "#1f77b4", "derived": "#ff7f0e", "implicit": "#2ca02c"}
    epoch_styles = {3: {"linestyle": "--", "marker": "o", "label_suffix": " (e3)"},
                     5: {"linestyle": "-",  "marker": "s", "label_suffix": " (e5)"}}

    # Plot one line per (variant, epoch_count)
    for v in variants:
        for e, sty in epoch_styles.items():
            ys = []
            for r in ranks:
                key = f"trained-heading-{v}_r{r}_e{e}"
                if key in summary:
                    ys.append(summary[key]["PASS"] * 100)
                else:
                    ys.append(np.nan)
            ax.plot(ranks, ys, color=colors[v],
                    label=f"{v}{sty['label_suffix']}",
                    linestyle=sty["linestyle"], marker=sty["marker"],
                    markersize=7, linewidth=1.8)

    # Add horizontal dashed lines for zero-shot baselines
    for v in variants:
        zs = summary.get(f"zs-heading-{v}", {}).get("PASS")
        if zs is not None:
            ax.axhline(zs * 100, color=colors[v], linestyle=":",
                       linewidth=1, alpha=0.5)
            ax.text(16.3, zs * 100, f"zs-{v}", color=colors[v],
                    fontsize=8, va="center", ha="left")

    ax.set_xlabel("LoRA rank", fontsize=11)
    ax.set_ylabel("PASS rate (%)", fontsize=11)
    ax.set_title("Rank-saturation: LoRA rank vs PASS, per variant × epoch count",
                 fontsize=12, pad=10)
    ax.set_xticks(ranks)
    ax.set_xticklabels(["r=4", "r=8", "r=16"])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


# ───────────── Figure 2: zero-shot vs best-trained bar chart ───────────────
def fig2_zs_vs_trained(summary, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    variants = ["given", "derived", "implicit"]

    zs_vals = []
    tr_vals = []
    tr_labels = []
    for v in variants:
        zs = summary.get(f"zs-heading-{v}", {}).get("PASS", 0) * 100
        # Find best trained PASS across r∈{4,8,16}, e∈{3,5}
        best = -1
        best_key = None
        for r in [4, 8, 16]:
            for e in [3, 5]:
                key = f"trained-heading-{v}_r{r}_e{e}"
                if key in summary:
                    p = summary[key]["PASS"] * 100
                    if p > best:
                        best, best_key = p, key
        zs_vals.append(zs)
        tr_vals.append(best)
        # Pretty label like "r=16 e5"
        m = re.search(r"_r(\d+)_e(\d+)", best_key or "")
        if m:
            tr_labels.append(f"r={m.group(1)} e{m.group(2)}")
        else:
            tr_labels.append("?")

    x = np.arange(len(variants))
    w = 0.35
    bars_zs = ax.bar(x - w/2, zs_vals, w, label="zero-shot",
                      color="#bbbbbb", edgecolor="black", linewidth=0.6)
    bars_tr = ax.bar(x + w/2, tr_vals, w, label="best LoRA",
                      color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                      edgecolor="black", linewidth=0.6)

    # Annotate bars with values
    for b, v in zip(bars_zs, zs_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9)
    for b, v, lbl in zip(bars_tr, tr_vals, tr_labels):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width()/2, v / 2, lbl,
                ha="center", fontsize=8, color="white", fontweight="bold")

    # Delta arrows
    for i, (zs, tr) in enumerate(zip(zs_vals, tr_vals)):
        ax.annotate("", xy=(i + w/2, tr), xytext=(i - w/2, zs),
                    arrowprops=dict(arrowstyle="->", color="red",
                                     alpha=0.4, lw=1))
        ax.text(i, max(zs, tr) + 6, f"+{tr - zs:.1f}pp",
                ha="center", fontsize=10, color="red", fontweight="bold")

    ax.set_ylabel("PASS rate (%)", fontsize=11)
    ax.set_title("Zero-shot vs best LoRA-trained: Qwen 2.5 VL 7B on Zurich navigation",
                 fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["heading-given", "heading-derived", "heading-implicit"],
                        fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


# ──────────── Figure 3: derived heading-inference scatter ──────────────────
def fig3_heading_scatter(summary, out_path: Path):
    """Predicted heading (parsed from "facing X°") vs GT heading.
    Shows 2 conditions side by side: zs vs trained-derived-r16-e5 (best).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300, sharey=True)

    panels = [
        ("zs-heading-derived",                   "Zero-shot Qwen"),
        ("trained-heading-derived_r16_e5",       "Trained (r=16, e=5)"),
    ]

    for ax, (key, title) in zip(axes, panels):
        rows = summary[key]["rows"]
        gt, pred, err = [], [], []
        for r in rows:
            resp = r.get("model_response", "")
            t1 = resp.find("<thinking>"); t2 = resp.find("</thinking>", t1)
            if t1 < 0: continue
            thinking = resp[t1+10: t2 if t2 > t1 else None]
            m = HEAD_RE.search(thinking)
            if not m: continue
            try: p = float(m.group(1))
            except ValueError: continue
            g = r.get("heading")
            if g is None: continue
            e = abs(((p - g + 180) % 360) - 180)
            gt.append(g); pred.append(p); err.append(e)
        gt = np.array(gt); pred = np.array(pred); err = np.array(err)
        n = len(gt)
        n_total = summary[key]["n"]

        # Identity reference line
        ax.plot([0, 360], [0, 360], "k--", alpha=0.4, linewidth=1)
        # 22.5° error bands (the verb-flip threshold)
        for delta in [22.5, -22.5]:
            ax.plot([0, 360], [delta, 360+delta], "k:", alpha=0.2,
                    linewidth=0.8)

        colors = np.where(err < 22.5, "tab:green",
                          np.where(err < 90, "tab:orange", "tab:red"))
        ax.scatter(gt, pred, c=colors, s=18, alpha=0.65, edgecolors="none")

        # Annotate
        within_22 = (err < 22.5).sum() / n * 100
        mean_err = err.mean()
        emit_pct = n / n_total * 100
        ax.text(0.05, 0.95,
                f"n={n}/{n_total} emit ({emit_pct:.0f}%)\n"
                f"within 22.5°: {within_22:.1f}%\n"
                f"mean |err|: {mean_err:.1f}°",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.3", alpha=0.85))

        ax.set_xlim(0, 360); ax.set_ylim(0, 360)
        ax.set_xlabel("GT heading (°)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Predicted heading (°)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    handles = [
        plt.Line2D([0],[0], marker="o", color="w",
                    markerfacecolor="tab:green", markersize=8, label="< 22.5° (verb-correct)"),
        plt.Line2D([0],[0], marker="o", color="w",
                    markerfacecolor="tab:orange", markersize=8, label="22.5-90° (close-ish)"),
        plt.Line2D([0],[0], marker="o", color="w",
                    markerfacecolor="tab:red", markersize=8, label="> 90° (wildly off)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
                bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    fig.suptitle("Heading inference: model's 'facing X°' vs GT (derived conditions)",
                  fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default="eval_pull/ablation_20260602_054707",
                    help="dir containing scored eval outputs")
    ap.add_argument("--out-dir", default="docs/figures",
                    help="where to write the 3 PNG figures")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(run_dir)
    print(f"loaded {len(summary)} conditions from {run_dir}")

    fig1_rank_saturation(summary, out_dir / "fig1_rank_saturation.png")
    fig2_zs_vs_trained(summary,   out_dir / "fig2_zs_vs_trained.png")
    fig3_heading_scatter(summary, out_dir / "fig3_heading_scatter.png")

    print(f"\nAll 3 figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
