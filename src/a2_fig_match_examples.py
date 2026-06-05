"""Generate docs/figures/fig4_match_success_failure.png — a 2x2 grid
showing 2 successful matched-cohort frames (row 1) and 2 failure cases
(row 2) for the 'noisy localization' paragraph in the report.

Success = matched at attraction level with multiple coincidences.
Failure = matched at poi level only (just a generic street name).
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FRAMES_ROOT = Path("data/cities/zurich/frames")
GVG = Path("data/cities/zurich/a2/GPS_VLM_GEO.jsonl")

CASES = [
    # (row, col, video, frame, label)
    (0, 0, "hidden_streets", "frame_01894", "SUCCESS"),
    (0, 1, "bahnhofstrasse", "frame_01412", "SUCCESS"),
    (1, 0, "bahnhofstrasse", "frame_01593", "FAILURE"),
    (1, 1, "hidden_streets", "frame_01633", "FAILURE"),
]

# Load match info from GPS_VLM_GEO
lookup = {}
for line in GVG.open(encoding="utf-8"):
    r = json.loads(line)
    lookup[(r["video"], r["frame_id"])] = r

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300)

for row, col, vid, fid, label in CASES:
    ax = axes[row, col]
    img_path = FRAMES_ROOT / vid / f"{fid}.jpg"
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])

    # Get match info
    r = lookup.get((vid, fid), {})
    matches = r.get("matches", [])
    best = r.get("best_level", "?")
    landmark_names = [m["vlm_name"] for m in matches]
    landmark_str = ", ".join(landmark_names[:3])
    if len(landmark_names) > 3:
        landmark_str += f", +{len(landmark_names)-3}"

    # Status badge color
    color = "tab:green" if label == "SUCCESS" else "tab:red"
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="white", va="top",
            bbox=dict(facecolor=color, edgecolor="none", pad=4))
    # Frame ID at top right
    ax.text(0.98, 0.97, f"{vid}/{fid}", transform=ax.transAxes,
            fontsize=8, color="white", va="top", ha="right",
            bbox=dict(facecolor="black", edgecolor="none", pad=2, alpha=0.6))

    # Caption below image
    caption = f"best_level: {best}   |   {len(matches)} match{'es' if len(matches)!=1 else ''}: {landmark_str}"
    ax.set_xlabel(caption, fontsize=9, color="black")

# Overall title
fig.suptitle(
    "Matched-cohort QC: successful matches (top) vs. noisy-localization failures (bottom)",
    fontsize=12, y=0.99)

# Subtitle row labels on the LEFT
fig.text(0.02, 0.72, "SUCCESS",
         fontsize=11, fontweight="bold", color="tab:green",
         rotation=90, va="center")
fig.text(0.02, 0.30, "FAILURE",
         fontsize=11, fontweight="bold", color="tab:red",
         rotation=90, va="center")

plt.tight_layout(rect=[0.03, 0, 1, 0.96])
out_path = Path("docs/figures/fig4_match_success_failure.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"saved {out_path}")
