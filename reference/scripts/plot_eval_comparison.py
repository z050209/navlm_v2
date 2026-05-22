"""Render bar charts comparing the 6 (or partial) experiment conditions.

Reads `results/eval_v3_<tag>.json` for each tag and produces:
  - results/plot_gate_pass_rates.png    grouped bar: 6 gates × 6 conditions
  - results/plot_pass_strict.png         single bar: PASS_strict per condition
  - results/plot_delta_distribution.png  stacked bar: <30 / 30-55 / >55 / no_verb
  - (if both v3-base and v3-lora exist) results/plot_lora_lift.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# Canonical column order
CONDITIONS = [
    ("base_v3",   "A1\nbase\n+v3 (hd)"),
    ("base_v4a",  "A2\nbase\n+v4a (no hd)"),
    ("base_v4b",  "A3\nbase\n+v4b (no hd, CoT)"),
    ("lora_v3",   "C1\nLoRA\n+v3"),
    ("lora_v4a",  "C2\nLoRA\n+v4a"),
    ("lora_v4b",  "C3\nLoRA\n+v4b"),
    ("lora_v4c",  "C4\nLoRA\n+v4c (Claude)"),
]
GATES = ["1_format", "2_sentence_count", "3_closed_loop",
         "4_checkpoint", "5_dest_correct", "6_anchor_grounded"]
GATE_LABELS = ["format", "sentences", "closed-loop", "checkpoint",
               "dest correct", "anchor grounded"]


def load_results():
    out = {}
    for tag, label in CONDITIONS:
        path = RESULTS / f"eval_v3_{tag}.json"
        if path.exists():
            out[tag] = (label, json.load(open(path)))
    return out


def plot_gate_pass_rates(data, out_path):
    tags = list(data.keys())
    n_tags = len(tags)
    n_gates = len(GATES)
    bar_w = 0.8 / n_tags
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.tab10(np.linspace(0, 1, n_tags))
    x = np.arange(n_gates)
    for i, tag in enumerate(tags):
        label, summary = data[tag]
        rates = [summary["gate_pass_rate"][g] * 100 for g in GATES]
        offset = (i - n_tags / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, rates, bar_w, label=label.replace("\n", " "),
                       color=cmap[i])
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.0f}",
                     ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(GATE_LABELS, fontsize=10)
    ax.set_ylabel("Pass rate (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Gate-by-gate pass rates across experimental conditions",
                  fontsize=13, weight="bold")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def plot_pass_strict(data, out_path):
    tags = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [data[t][0] for t in tags]
    values = [data[t][1]["pass_strict_30"] * 100 for t in tags]
    cmap = plt.cm.tab10(np.linspace(0, 1, len(tags)))
    bars = ax.bar(labels, values, color=cmap, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("PASS_strict (<30°) % of samples", fontsize=11)
    ax.set_title("Overall pass rate (all 6 gates AND closed-loop δ < 30°)",
                  fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def plot_closed_loop_alone(data, out_path):
    """Gate 3 only — clean signal of geometric correctness."""
    tags = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [data[t][0] for t in tags]
    values = [data[t][1]["gate_pass_rate"]["3_closed_loop"] * 100 for t in tags]
    cmap = plt.cm.tab10(np.linspace(0, 1, len(tags)))
    bars = ax.bar(labels, values, color=cmap, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=10, weight="bold")
    ax.axhline(y=25, color="grey", linestyle="--", alpha=0.5,
                label="random 4-action baseline (25%)")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Closed-loop pass rate (%)", fontsize=11)
    ax.set_title("Geometric correctness — does the action verb point the right way?",
                  fontsize=13, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def plot_delta_distribution(data, out_path):
    tags = list(data.keys())
    labels = [data[t][0] for t in tags]
    bins = ["<30°", "30-55°", ">55°"]
    bin_data = []
    for t in tags:
        d = data[t][1]["delta_distribution"]
        lt30 = d["lt_30"]
        lt55 = d["lt_55"]
        bin_data.append([lt30, lt55 - lt30, 1 - lt55])
    bin_data = np.array(bin_data) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4caf50", "#ffc107", "#f44336"]
    bottom = np.zeros(len(tags))
    for i, b in enumerate(bins):
        vals = bin_data[:, i]
        ax.bar(labels, vals, bottom=bottom, label=b, color=colors[i],
                edgecolor="black", linewidth=0.5)
        for j, v in enumerate(vals):
            if v > 5:
                ax.text(j, bottom[j] + v / 2, f"{v:.0f}%",
                         ha="center", va="center", fontsize=9, weight="bold",
                         color="white" if i != 1 else "black")
        bottom += vals
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of samples", fontsize=11)
    ax.set_title("Closed-loop δ distribution (geometric error band)",
                  fontsize=13, weight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=RESULTS)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_results()
    if not data:
        print("no eval_v3_*.json found")
        return
    print(f"loaded {len(data)} conditions: {list(data.keys())}")

    plot_gate_pass_rates(data, out_dir / "plot_gate_pass_rates.png")
    plot_pass_strict(data, out_dir / "plot_pass_strict.png")
    plot_closed_loop_alone(data, out_dir / "plot_closed_loop.png")
    plot_delta_distribution(data, out_dir / "plot_delta_distribution.png")


if __name__ == "__main__":
    main()
