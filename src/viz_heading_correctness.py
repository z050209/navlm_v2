"""Two PNGs explaining the heading-correctness check used at
(a) annotation time and (b) eval time.

Both PNGs visualize the SAME closed-loop verifier math:

    δ = | angle_diff( heading + ACTION_DELTA[verb], route_bearing ) |
    sample is correct iff verb is recognised AND δ < 30°.

The difference is only WHERE the verb comes from:

  (a)  annotation time:  verb comes from the TEACHER (Gemini 2.5 Pro).
                         A failed verifier DROPS the (frame, dest)
                         sample from the training set.
  (b)  eval time:        verb comes from the MODEL UNDER TEST (the
                         zero-shot baselines or the LoRA adapters).
                         A failed verifier fails the
                         "directional_accuracy" metric — one of the
                         4 metrics ANDed into PASS_strict.

Outputs (saved to viz/):

  heading_filter_annotation.png   — annotation-time filter, 4 panels
                                    (2 PASS + 2 FAIL real examples)
  heading_filter_eval.png         — eval-time scoring, same layout
                                    with the source labels changed

  python -m src.viz_heading_correctness
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402

ACTION_DELTA = {"continue ahead": 0.0, "turn left": -90.0,
                "turn right": 90.0, "turn around": 180.0}


def _angle_diff(a, b):
    return ((a - b + 540) % 360) - 180


def _polar_arrow(ax, bearing_deg, length=0.85, **kw):
    """Draw an arrow from origin pointing in compass-bearing direction
    (0°=N up, clockwise). Length in axis units."""
    rad = math.radians(bearing_deg)
    # convert compass (N up, CW) to mpl xy (E right, CCW from x-axis):
    # x = sin(bearing), y = cos(bearing)
    x = length * math.sin(rad)
    y = length * math.cos(rad)
    ax.annotate("", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,"
                                "head_length=0.5",
                                lw=2.2, **kw))


def _compass_panel(ax, frame, title, source_label_h, source_label_v,
                   pass_fail_text):
    """Render one compass panel showing the 3 arrows + δ + PASS/FAIL."""
    h = frame["heading"]
    B = frame["route_bearing"]
    verb = frame["action"]
    delta_v = ACTION_DELTA.get(verb, 0.0)
    new_h = (h + delta_v) % 360
    delta = abs(_angle_diff(new_h, B))
    accepted = frame["accepted"]

    # circle + cardinals
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    theta = [i * math.pi / 180 for i in range(361)]
    ax.plot([math.sin(t) for t in theta], [math.cos(t) for t in theta],
            color="#999", lw=0.8)
    for angle, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        r = math.radians(angle)
        ax.text(1.13 * math.sin(r), 1.13 * math.cos(r),
                label, ha="center", va="center", fontsize=10,
                color="#444")

    # the three arrows (drawn from origin)
    _polar_arrow(ax, h, length=0.78, color="#1f77b4")            # camera (h)
    _polar_arrow(ax, B, length=0.78, color="#2ca02c")            # route bearing (B)
    _polar_arrow(ax, new_h, length=0.55, color="#d62728")        # h + ACTION_DELTA[verb]

    # arc showing δ between new_h and B
    a0 = min(new_h, B)
    a1 = max(new_h, B)
    if a1 - a0 > 180:
        a0, a1 = a1, a0 + 360
    arc_thetas = [math.radians(a) for a in
                  [a0 + i * (a1 - a0) / 40 for i in range(41)]]
    rr = 0.35
    ax.plot([rr * math.sin(t) for t in arc_thetas],
            [rr * math.cos(t) for t in arc_thetas],
            color="#d62728", lw=1.4)
    mid = math.radians((a0 + a1) / 2)
    ax.text(0.50 * math.sin(mid), 0.50 * math.cos(mid),
            f"δ={delta:.0f}°", color="#d62728", fontsize=11,
            ha="center", va="center", fontweight="bold")

    # text legend top-left of panel
    box = ("FILTER FED:\n"
           f"  • heading (h) = {h:.0f}°  ({source_label_h})\n"
           f"  • route_bearing (B) = {B:.0f}°  (OSM walking-graph)\n"
           f"  • verb = '{verb}'  ({source_label_v})\n"
           "\n"
           "CHECK:\n"
           f"  h + ACTION_DELTA[verb] = {h:.0f} + {delta_v:+.0f} "
           f"= {new_h:.0f}°\n"
           f"  δ = |{new_h:.0f}° − {B:.0f}°| = {delta:.0f}°")
    ax.text(-1.25, -1.25, box, ha="left", va="bottom", fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#f8f8f8", edgecolor="#aaa"))

    # PASS/FAIL stamp
    color = "#2a9d8f" if accepted else "#c0392b"
    ax.text(0, 1.22, pass_fail_text, ha="center", va="center",
            fontsize=11, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor=color, lw=1.5))

    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_global_legend(fig):
    """Three-arrow legend in the figure's bottom margin."""
    import matplotlib.patches as mpatches
    handles = [
        mpatches.FancyArrow(0, 0, 0, 0, color="#1f77b4",
                            label="heading h  (recovered camera direction)"),
        mpatches.FancyArrow(0, 0, 0, 0, color="#2ca02c",
                            label="route_bearing B  (first OSM walking segment)"),
        mpatches.FancyArrow(0, 0, 0, 0, color="#d62728",
                            label="h + ACTION_DELTA[verb]  (where the user "
                                  "would face AFTER the verb)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))


def _pick_examples(rows, want_pass=2, want_fail=2):
    """Pick examples spanning the 4 verbs if possible."""
    by_verb_pass = {v: [] for v in ACTION_DELTA}
    by_verb_fail = {v: [] for v in ACTION_DELTA}
    for r in rows:
        v = r.get("action")
        if v not in ACTION_DELTA:
            continue
        d = r.get("verifier_delta")
        if d is None:
            continue
        (by_verb_pass if r.get("accepted") else by_verb_fail)[v].append(r)

    passed, failed = [], []
    # diverse-verb sampling
    for v in ["continue ahead", "turn right", "turn left", "turn around"]:
        if by_verb_pass[v] and len(passed) < want_pass:
            passed.append(by_verb_pass[v][0])
        if by_verb_fail[v] and len(failed) < want_fail:
            failed.append(by_verb_fail[v][0])
    # backfill if we couldn't find enough variety
    flat_p = [r for v in ACTION_DELTA for r in by_verb_pass[v]]
    flat_f = [r for v in ACTION_DELTA for r in by_verb_fail[v]]
    while len(passed) < want_pass and flat_p:
        for r in flat_p:
            if r not in passed:
                passed.append(r); break
    while len(failed) < want_fail and flat_f:
        for r in flat_f:
            if r not in failed:
                failed.append(r); break
    return passed[:want_pass], failed[:want_fail]


def _render(rows, out_path, fig_title, source_label_h,
            source_label_v_for_kept, source_label_v_for_failed,
            pass_text, fail_text):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    passed, failed = _pick_examples(rows, want_pass=2, want_fail=2)
    if len(passed) < 2 or len(failed) < 2:
        print(f"[viz_heading_correctness] warning: only "
              f"{len(passed)} passes / {len(failed)} fails available;"
              f" panels will be sparse")

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.96)
    fig.text(0.5, 0.925,
             "Closed-loop verifier: a sample is CORRECT iff "
             "|angle_diff(heading + ACTION_DELTA[verb], "
             "route_bearing)| < 30°",
             ha="center", fontsize=10, style="italic", color="#333")

    panels = [
        (axes[0][0], passed[0] if passed else None,
         f"PASS · {passed[0]['video']}/{passed[0]['frame_id']}"
         if passed else "PASS (none)", source_label_v_for_kept, pass_text),
        (axes[0][1], passed[1] if len(passed) > 1 else None,
         f"PASS · {passed[1]['video']}/{passed[1]['frame_id']}"
         if len(passed) > 1 else "PASS (only 1 found)",
         source_label_v_for_kept, pass_text),
        (axes[1][0], failed[0] if failed else None,
         f"FAIL · {failed[0]['video']}/{failed[0]['frame_id']}"
         if failed else "FAIL (none)",
         source_label_v_for_failed, fail_text),
        (axes[1][1], failed[1] if len(failed) > 1 else None,
         f"FAIL · {failed[1]['video']}/{failed[1]['frame_id']}"
         if len(failed) > 1 else "FAIL (only 1 found)",
         source_label_v_for_failed, fail_text),
    ]
    for ax, frame, title, src_v, badge in panels:
        if frame is None:
            ax.set_visible(False)
            continue
        _compass_panel(ax, frame, title, source_label_h, src_v, badge)

    _add_global_legend(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_heading_correctness] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "annotations_strict.jsonl"))
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")
    rows = [json.loads(l) for l in in_path.open(encoding="utf-8")
            if l.strip()]
    print(f"[viz_heading_correctness] loaded {len(rows)} rows from "
          f"{in_path.name}")

    # PNG 1 — annotation-time filter
    _render(
        rows,
        config.VIZ_DIR / "heading_filter_annotation.png",
        fig_title=("(a) Annotation-time filter — drop a "
                   "(frame, destination) sample if Pro's verb doesn't "
                   "point the right way"),
        source_label_h="from trusted_frames.jsonl, recovered by DINOv2",
        source_label_v_for_kept="from Gemini 2.5 Pro · KEPT for training",
        source_label_v_for_failed="from Gemini 2.5 Pro · DROPPED from training",
        pass_text="KEPT (δ < 30°)",
        fail_text="DROPPED (δ ≥ 30°)")

    # PNG 2 — eval-time scoring
    _render(
        rows,
        config.VIZ_DIR / "heading_filter_eval.png",
        fig_title=("(b) Eval-time scoring — directional_accuracy metric "
                   "(1 of 4 ANDed into PASS_strict)"),
        source_label_h="from the test frame's GPS recovery (ground truth)",
        source_label_v_for_kept="from the model under test · CORRECT (counts toward PASS)",
        source_label_v_for_failed="from the model under test · WRONG (fails directional_accuracy)",
        pass_text="CORRECT (δ < 30°)",
        fail_text="WRONG (δ ≥ 30°)")


if __name__ == "__main__":
    main()
