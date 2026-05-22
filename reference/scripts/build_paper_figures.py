"""Build paper-ready figures into draft/pic.

This script generates:
  - teaser.png
  - gps_pipeline.png
  - instruction_pipeline.png
  - plot_pass_strict.png
  - plot_gate_pass_rates.png
  - plot_delta_distribution.png
  - plot_closed_loop.png
  - plot_heading_error.png

The experiment plots are intentionally restricted to the six conditions
that appear in the current paper draft (A1-A3, C1-C3), even though the
repository also contains a C4 evaluation.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "draft" / "pic"
TEASER_SAMPLE_FRAME = OUTPUT_DIR / "source" / "teaser_real_sample.jpg"
TEASER_SAMPLE_JSONL = ROOT / "data" / "cities" / "zurich" / "synth_v4b_train.jsonl"
TEASER_SAMPLE_IMAGE_KEY = (
    "/frames/extra_Switzerland_Zurich_Bahnhofstrasse_Walking_tour_Cit/frame_00294.jpg"
)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 160,
    }
)


@dataclass(frozen=True)
class Condition:
    tag: str
    short: str
    label: str
    color: str


CONDITIONS = [
    Condition("base_v3", "A1", "A1\nbase\ngiven", "#B0BEC5"),
    Condition("base_v4a", "A2", "A2\nbase\nhidden", "#CFD8DC"),
    Condition("base_v4b", "A3", "A3\nbase\ninfer", "#ECEFF1"),
    Condition("lora_v3", "C1", "C1\nLoRA\ngiven", "#2E7D32"),
    Condition("lora_v4a", "C2", "C2\nLoRA\nhidden", "#66BB6A"),
    Condition("lora_v4b", "C3", "C3\nLoRA\ninfer", "#A5D6A7"),
]

GATES = [
    ("1_format", "Fmt"),
    ("2_sentence_count", "Len"),
    ("3_closed_loop", "Closed"),
    ("4_checkpoint", "Checkpoint"),
    ("5_dest_correct", "Dest"),
    ("6_anchor_grounded", "Anchor"),
]

HEADING_RE = re.compile(r"INFERRED_HEADING:\s*(-?\d+(?:\.\d+)?)")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_summaries() -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for cond in CONDITIONS:
        path = RESULTS_DIR / f"eval_v3_{cond.tag}.json"
        summaries[cond.tag] = load_json(path)
    return summaries


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def angular_error(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def parsed_heading_errors() -> list[float]:
    rows = load_jsonl(RESULTS_DIR / "eval_v3_lora_v4b.jsonl")
    errors = []
    for row in rows:
        match = HEADING_RE.search(row.get("model_response", ""))
        if not match:
            continue
        inferred = float(match.group(1))
        gt = float(row["heading_gt"])
        errors.append(angular_error(inferred, gt))
    return errors


def delta_samples(tag: str) -> list[float]:
    rows = load_jsonl(RESULTS_DIR / f"eval_v3_{tag}.jsonl")
    deltas = []
    for row in rows:
        value = row.get("delta")
        if value is None:
            continue
        deltas.append(abs(float(value)))
    return deltas


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.sort(np.asarray(values, dtype=float))
    rank = min(len(arr) - 1, max(0, math.ceil(q * len(arr)) - 1))
    return float(arr[rank])


def save(fig: plt.Figure, name: str) -> None:
    out = OUTPUT_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str = "#334155",
    fontsize: int = 11,
    weight: str = "normal",
    radius: float = 0.03,
    ha: str = "center",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if ha == "center" else x + 0.03
    ax.text(
        tx,
        y + h / 2,
        text,
        ha=ha,
        va="center",
        fontsize=fontsize,
        weight=weight,
        multialignment=ha,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#475569") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


def _titled_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str = "#334155",
    title_color: str = "#0F172A",
    body_color: str = "#1E293B",
    title_size: float = 9.6,
    body_size: float = 9.3,
    body_offset: float = 0.07,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    title_text = ax.text(
        x + 0.03,
        y + h - 0.03,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        weight="bold",
        color=title_color,
        clip_on=True,
    )
    body_text = ax.text(
        x + 0.03,
        y + h - body_offset,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=body_color,
        linespacing=1.16,
        clip_on=True,
    )
    title_text.set_clip_path(patch)
    body_text.set_clip_path(patch)


def _crop_center_to_aspect(image: np.ndarray, target_aspect: float) -> np.ndarray:
    h, w = image.shape[:2]
    current_aspect = w / h
    if abs(current_aspect - target_aspect) < 1e-3:
        return image
    if current_aspect > target_aspect:
        new_w = max(1, int(round(h * target_aspect)))
        left = max(0, (w - new_w) // 2)
        return image[:, left : left + new_w]
    new_h = max(1, int(round(w / target_aspect)))
    top = max(0, (h - new_h) // 2)
    return image[top : top + new_h, :]


def _rounded_panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str,
    edge: str = "#D8E1F0",
    radius: float = 0.03,
    linewidth: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    return patch


def _pill(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str,
    text_color: str,
    edge: str | None = None,
    fontsize: float = 8.0,
    weight: str = "bold",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        linewidth=0.8 if edge else 0.0,
        edgecolor=edge or face,
        facecolor=face,
    )
    ax.add_patch(patch)
    text_obj = ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=text_color,
        clip_on=True,
    )
    text_obj.set_clip_path(patch)
    return patch


def _reference_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    body: str,
    *,
    face: str,
    pill_face: str,
    pill_text: str,
    edge: str = "#D8E1F0",
    body_color: str = "#475569",
    body_size: float = 9.0,
    align: str = "left",
    body_top_pad: float = 0.092,
    pill_width: float | None = None,
) -> FancyBboxPatch:
    patch = _rounded_panel(ax, x, y, w, h, face=face, edge=edge, radius=0.026, linewidth=0.95)
    pill_w = pill_width if pill_width is not None else min(w - 0.05, max(0.10, 0.018 * len(label) + 0.05))
    _pill(ax, x + 0.022, y + h - 0.052, pill_w, 0.034, label, face=pill_face, text_color=pill_text)
    tx = x + 0.026 if align == "left" else x + w / 2
    text_obj = ax.text(
        tx,
        y + h - body_top_pad,
        body,
        ha=align,
        va="top",
        fontsize=body_size,
        color=body_color,
        linespacing=1.16,
        clip_on=True,
    )
    text_obj.set_clip_path(patch)
    return patch


def _summary_strip(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    items: list[tuple[str, str]],
    *,
    face: str = "white",
    edge: str = "#D8E1F0",
    title_color: str = "#334155",
    body_color: str = "#64748B",
) -> FancyBboxPatch:
    patch = _rounded_panel(ax, x, y, w, h, face=face, edge=edge, radius=0.024, linewidth=0.95)
    n = len(items)
    for idx, (value, label) in enumerate(items):
        cx = x + w * (idx + 0.5) / n
        if idx:
            ax.plot([x + w * idx / n, x + w * idx / n], [y + 0.015, y + h - 0.015], color=edge, linewidth=0.8)
        value_text = ax.text(cx, y + h * 0.62, value, ha="center", va="center", fontsize=10.2, weight="bold", color=title_color, clip_on=True)
        label_text = ax.text(cx, y + h * 0.30, label, ha="center", va="center", fontsize=8.1, color=body_color, clip_on=True)
        value_text.set_clip_path(patch)
        label_text.set_clip_path(patch)
    return patch


def extract_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return text.strip()


def teaser_answer_preview(answer: str) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer.strip()) if part.strip()]
    if not parts:
        return answer.strip()
    return " ".join(parts[:2])


def load_teaser_sample() -> dict[str, str]:
    for row in load_jsonl(TEASER_SAMPLE_JSONL):
        image_path = row.get("image", "")
        if not image_path.endswith(TEASER_SAMPLE_IMAGE_KEY):
            continue
        user_text = row["messages"][1]["content"]
        answer = extract_answer(row["messages"][2]["content"])
        lines = [line.rstrip() for line in user_text.splitlines()]
        gps_line = next((line.replace("You are at GPS ", "GPS ").strip(".") for line in lines if line.startswith("You are at GPS")), "")
        question = next((line.replace("User asks: ", "").strip() for line in lines if line.startswith("User asks:")), "")
        poi_names = []
        in_poi_block = False
        for line in lines:
            if line.startswith("Nearby POIs"):
                in_poi_block = True
                continue
            if line.startswith("OSM-planned walking route"):
                in_poi_block = False
            if in_poi_block and line.startswith("  - "):
                poi_names.append(line[4:].split(" (", 1)[0])
        bearing_match = re.search(r"First segment absolute bearing:\s*([0-9]+)", user_text)
        distance_match = re.search(r"Total distance:\s*([0-9]+)\s*m", user_text)
        route_bits = []
        if bearing_match:
            route_bits.append(f"first bearing {bearing_match.group(1)} deg")
        if distance_match:
            route_bits.append(f"distance {distance_match.group(1)} m")
        poi_preview = []
        if poi_names:
            poi_preview = [poi_names[0]]
        if len(poi_names) >= 2:
            poi_preview.append(poi_names[1])
        if len(poi_names) >= 3:
            poi_preview.append(poi_names[-1])
        poi_preview = list(dict.fromkeys(poi_preview))
        return {
            "image_path": image_path,
            "gps_line": gps_line,
            "route_line": ", ".join(route_bits),
            "question": question,
            "answer": answer,
            "poi_summary": ", ".join(poi_preview),
        }
    raise FileNotFoundError(f"Could not locate teaser sample matching {TEASER_SAMPLE_IMAGE_KEY}")


def draw_teaser() -> None:
    sample = load_teaser_sample()
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    edge = "#D8E1F0"
    title_color = "#334155"
    body_color = "#64748B"

    board = _rounded_panel(ax, 0.02, 0.05, 0.96, 0.90, face="#F4F7FD", edge=edge, radius=0.032, linewidth=1.0)
    title = ax.text(0.50, 0.90, "Training Example Used for Instruction Tuning", ha="center", va="center", fontsize=13.4, weight="bold", color=title_color, clip_on=True)
    subtitle = ax.text(
        0.50,
        0.865,
        "GPS, nearby POIs, and route metadata are shown to the model, but camera heading is withheld.",
        ha="center",
        va="center",
        fontsize=8.9,
        color=body_color,
        clip_on=True,
    )
    title.set_clip_path(board)
    subtitle.set_clip_path(board)

    image_panel = _rounded_panel(ax, 0.05, 0.16, 0.47, 0.66, face="white", edge=edge, radius=0.028, linewidth=0.95)
    _pill(ax, 0.07, 0.775, 0.17, 0.036, "Street-view frame", face="#DBEAFE", text_color="#1D4ED8")
    image = plt.imread(TEASER_SAMPLE_FRAME)
    target_aspect = (0.43 * fig.get_figwidth()) / (0.54 * fig.get_figheight())
    image = _crop_center_to_aspect(image, target_aspect)
    image_ax = ax.inset_axes([0.07, 0.205, 0.43, 0.54], transform=ax.transAxes)
    image_ax.imshow(image, aspect="auto")
    image_ax.axis("off")
    image_note = ax.text(0.285, 0.185, "The target turn must be inferred from the scene itself.", ha="center", va="center", fontsize=8.8, color=body_color, clip_on=True)
    image_note.set_clip_path(image_panel)

    panel = _rounded_panel(ax, 0.56, 0.16, 0.37, 0.66, face="white", edge=edge, radius=0.028, linewidth=0.95)
    _pill(ax, 0.58, 0.775, 0.18, 0.036, "Prompt fields shown", face="#DBEAFE", text_color="#1D4ED8")

    question_text = sample["question"].strip()
    if question_text.startswith('"') and question_text.endswith('"'):
        question_text = question_text[1:-1]
    poi_bits = sample["poi_summary"].split(", ")
    poi_brief = ", ".join(poi_bits[:2]) if len(poi_bits) >= 2 else sample["poi_summary"]
    route_brief = sample["route_line"].replace("first bearing ", "bearing ").replace("distance ", "")
    gps_brief = sample["gps_line"].replace("GPS (", "GPS ").replace(")", "")
    context_text = "\n".join(
        [
            textwrap.fill(gps_brief, width=36),
            textwrap.fill(f"POIs: {poi_brief}", width=36),
            textwrap.fill(f"Route: {route_brief}", width=36),
        ]
    )
    answer_preview = teaser_answer_preview(sample["answer"])
    answer_brief = answer_preview.split(",", 1)[0].strip()
    if answer_brief and answer_brief[-1] not in ".!?":
        answer_brief = f"{answer_brief}."
    answer_text = answer_brief

    _reference_card(
        ax,
        0.58,
        0.598,
        0.31,
        0.135,
        "Question",
        question_text,
        face="#FFF7DB",
        pill_face="#FDE68A",
        pill_text="#92400E",
        body_size=7.8,
        body_top_pad=0.088,
        pill_width=0.15,
    )
    _reference_card(
        ax,
        0.58,
        0.378,
        0.33,
        0.182,
        "Prompt context",
        context_text,
        face="#EDF4FF",
        pill_face="#DBEAFE",
        pill_text="#1D4ED8",
        body_size=8.0,
        body_top_pad=0.100,
        pill_width=0.19,
    )
    _reference_card(
        ax,
        0.58,
        0.188,
        0.33,
        0.132,
        "Target instruction",
        answer_text,
        face="#EAF8EF",
        pill_face="#BBF7D0",
        pill_text="#166534",
        body_size=8.0,
        body_top_pad=0.080,
        pill_width=0.22,
    )
    footer = _rounded_panel(ax, 0.05, 0.085, 0.88, 0.052, face="#E8F0FF", edge=edge, radius=0.022, linewidth=0.95)
    footer_text = ax.text(
        0.49,
        0.111,
        "Heading is hidden, so the model must infer the turn direction from the image.",
        ha="center",
        va="center",
        fontsize=8.7,
        color=title_color,
        clip_on=True,
    )
    footer_text.set_clip_path(footer)
    save(fig, "teaser.png")


def draw_gps_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 5.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    frame = plt.imread(TEASER_SAMPLE_FRAME)
    height, width = frame.shape[:2]
    match_thumb = frame[int(0.08 * height) : int(0.74 * height), : int(0.34 * width)]
    landmark_thumb = frame[int(0.10 * height) : int(0.72 * height), int(0.30 * width) : int(0.72 * width)]
    heading_thumb = frame[int(0.54 * height) : height, int(0.12 * width) : width]

    edge = "#D8E1F0"
    title_color = "#334155"
    body_color = "#475569"
    line_color = "#5E738E"

    board = _rounded_panel(ax, 0.02, 0.06, 0.96, 0.88, face="#F4F7FD", edge=edge, radius=0.032, linewidth=1.0)
    title = ax.text(0.50, 0.89, "GPS Recovery from Frame Evidence", ha="center", va="center", fontsize=13.2, weight="bold", color=title_color, clip_on=True)
    subtitle = ax.text(
        0.50,
        0.855,
        "Retrieved street matches, visible POIs, and road layout cues are combined before trusted starts are kept.",
        ha="center",
        va="center",
        fontsize=8.9,
        color=body_color,
        clip_on=True,
    )
    title.set_clip_path(board)
    subtitle.set_clip_path(board)

    def _icon(cx: float, cy: float, kind: str, color: str) -> None:
        radius = 0.023
        icon_ax = ax.inset_axes([cx - radius, cy - radius, radius * 2, radius * 2], transform=ax.transAxes)
        icon_ax.set_xlim(0, 1)
        icon_ax.set_ylim(0, 1)
        icon_ax.set_aspect("equal")
        icon_ax.axis("off")
        icon_ax.add_patch(Circle((0.5, 0.5), 0.48, facecolor=color, edgecolor="none"))
        if kind == "play":
            icon_ax.add_patch(Polygon([(0.40, 0.30), (0.40, 0.70), (0.72, 0.50)], closed=True, facecolor="white", edgecolor="white"))
        elif kind == "pin":
            icon_ax.add_patch(Circle((0.50, 0.60), 0.15, facecolor="white", edgecolor="white"))
            icon_ax.add_patch(Polygon([(0.50, 0.16), (0.34, 0.46), (0.66, 0.46)], closed=True, facecolor="white", edgecolor="white"))
        elif kind == "road":
            icon_ax.plot([0.24, 0.76], [0.26, 0.74], color="white", linewidth=2.2, solid_capstyle="round")
            icon_ax.plot([0.43, 0.57], [0.43, 0.57], color=color, linewidth=1.3, solid_capstyle="round")
        elif kind == "scan":
            icon_ax.add_patch(Circle((0.44, 0.56), 0.18, facecolor="none", edgecolor="white", linewidth=1.9))
            icon_ax.plot([0.58, 0.78], [0.42, 0.22], color="white", linewidth=2.0, solid_capstyle="round")
        elif kind == "check":
            icon_ax.plot([0.24, 0.42, 0.76], [0.48, 0.28, 0.68], color="white", linewidth=2.2, solid_capstyle="round")

    def _card(x: float, y: float, w: float, h: float, title: str, body: str, face: str, icon_color: str, icon_kind: str) -> None:
        patch = _rounded_panel(ax, x, y, w, h, face=face, edge=edge, radius=0.026, linewidth=0.95)
        _pill(ax, x + 0.055, y + h - 0.050, min(w - 0.08, max(0.12, 0.017 * len(title) + 0.045)), 0.034, title, face="white", text_color=title_color, edge=edge)
        _icon(x + 0.028, y + h - 0.045, icon_kind, icon_color)
        body_text = ax.text(x + 0.028, y + h - 0.100, body, ha="left", va="top", fontsize=9.2, color=body_color, linespacing=1.18, clip_on=True)
        body_text.set_clip_path(patch)

    def _thumb(x: float, y: float, w: float, h: float, image: np.ndarray, label: str) -> None:
        patch = _rounded_panel(ax, x, y, w, h, face="#FAFCFF", edge=edge, radius=0.020, linewidth=0.85)
        inset = ax.inset_axes([x + 0.009, y + 0.052, w - 0.018, h - 0.092], transform=ax.transAxes)
        inset.imshow(image)
        inset.axis("off")
        label_text = ax.text(x + w / 2, y + 0.023, label, ha="center", va="center", fontsize=8.5, color=body_color, clip_on=True)
        label_text.set_clip_path(patch)

    def _arrow(start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.4,
                color=line_color,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=2,
                shrinkB=2,
            )
        )

    left_panel = _rounded_panel(ax, 0.05, 0.17, 0.39, 0.61, face="white", edge=edge, radius=0.028, linewidth=0.95)
    right_panel = _rounded_panel(ax, 0.47, 0.17, 0.46, 0.61, face="white", edge=edge, radius=0.028, linewidth=0.95)
    _pill(ax, 0.07, 0.735, 0.17, 0.036, "Frame evidence", face="#DBEAFE", text_color="#1D4ED8")
    _pill(ax, 0.49, 0.735, 0.22, 0.036, "Localization decisions", face="#DBEAFE", text_color="#1D4ED8")

    _summary_strip(
        ax,
        0.07,
        0.615,
        0.34,
        0.085,
        [("8", "Zurich walks"), ("27,075", "frames"), ("1 fps", "sampling")],
        face="#F8FAFF",
        edge=edge,
    )
    _thumb(0.078, 0.33, 0.105, 0.22, match_thumb, "street match")
    _thumb(0.196, 0.33, 0.105, 0.22, landmark_thumb, "storefront")
    _thumb(0.314, 0.33, 0.105, 0.22, heading_thumb, "street layout")
    evidence_note = ax.text(0.245, 0.255, "One frame provides three\ncomplementary localization cues.", ha="center", va="center", fontsize=9.0, color=body_color, linespacing=1.18, clip_on=True)
    evidence_note.set_clip_path(left_panel)

    _card(0.50, 0.49, 0.19, 0.18, "Candidate GPS", "street match +\nlandmarks + heading", "#EAF4FF", "#38BDF8", "pin")
    _card(0.71, 0.49, 0.19, 0.18, "Road snap", "align to the OSM\nwalking graph", "#EDF4FF", "#60A5FA", "road")
    _card(0.50, 0.25, 0.19, 0.18, "Visible-POI check", "visible POIs\nmust agree", "#F3ECFF", "#A78BFA", "scan")
    _card(0.71, 0.25, 0.19, 0.18, "Trusted starts", "2,177 frames\nkept", "#EAF8EF", "#4ADE80", "check")

    _arrow((0.44, 0.52), (0.50, 0.58))
    _arrow((0.44, 0.42), (0.50, 0.34))
    _arrow((0.69, 0.58), (0.71, 0.58))
    _arrow((0.805, 0.49), (0.805, 0.43))
    _arrow((0.69, 0.34), (0.71, 0.34))

    footer = _rounded_panel(ax, 0.11, 0.085, 0.75, 0.058, face="#E8F0FF", edge=edge, radius=0.022, linewidth=0.95)
    footer_text = ax.text(0.485, 0.114, "27,075 video frames yield 2,177 trusted starts after map snapping and POI checks.", ha="center", va="center", fontsize=9.0, color=title_color, clip_on=True)
    footer_text.set_clip_path(footer)
    save(fig, "gps_pipeline.png")


def draw_instruction_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 5.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    edge = "#D8E1F0"
    title_color = "#334155"
    body_color = "#475569"
    line_color = "#5E738E"

    board = _rounded_panel(ax, 0.02, 0.06, 0.96, 0.88, face="#F4F7FD", edge=edge, radius=0.032, linewidth=1.0)
    title = ax.text(0.50, 0.89, "From Trusted Frames to Instruction-Tuning Tuples", ha="center", va="center", fontsize=13.2, weight="bold", color=title_color, clip_on=True)
    subtitle = ax.text(
        0.50,
        0.855,
        "Each trusted frame becomes route-conditioned teacher supervision before strict filtering and LoRA training.",
        ha="center",
        va="center",
        fontsize=8.9,
        color=body_color,
        clip_on=True,
    )
    title.set_clip_path(board)
    subtitle.set_clip_path(board)

    def _badge(cx: float, cy: float, color: str, glyph: str) -> None:
        radius = 0.022
        badge_ax = ax.inset_axes([cx - radius, cy - radius, radius * 2, radius * 2], transform=ax.transAxes)
        badge_ax.set_xlim(0, 1)
        badge_ax.set_ylim(0, 1)
        badge_ax.set_aspect("equal")
        badge_ax.axis("off")
        badge_ax.add_patch(Circle((0.5, 0.5), 0.48, facecolor=color, edgecolor="none"))
        badge_ax.text(0.5, 0.5, glyph, ha="center", va="center", fontsize=10.5, weight="bold", color="white")

    def _card(x: float, y: float, w: float, h: float, title: str, body: str, face: str, badge_color: str, glyph: str) -> None:
        patch = _rounded_panel(ax, x, y, w, h, face=face, edge=edge, radius=0.024, linewidth=0.95)
        # Pill width grows with title length; capped to fit inside card minus left badge + right padding.
        pill_w = max(0.082, 0.011 * len(title) + 0.030)
        pill_w = min(pill_w, w - 0.062)
        _pill(ax, x + 0.052, y + h - 0.046, pill_w, 0.032, title, face="white", text_color=title_color, edge=edge, fontsize=7.6)
        _badge(x + 0.034, y + h - 0.041, badge_color, glyph)
        body_text = ax.text(x + 0.024, y + h - 0.089, body, ha="left", va="top", fontsize=8.1, color=body_color, linespacing=1.18, clip_on=True)
        body_text.set_clip_path(patch)

    def _arrow(start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.35,
                color=line_color,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=2,
                shrinkB=2,
            )
        )

    _summary_strip(
        ax,
        0.09,
        0.72,
        0.82,
        0.09,
        [("2,177", "trusted frames"), ("6,510", "teacher answers"), ("6,249", "valid after checks"), ("4,689", "strict tuples")],
        face="white",
        edge=edge,
    )

    route_panel = _rounded_panel(ax, 0.05, 0.44, 0.57, 0.21, face="white", edge=edge, radius=0.026, linewidth=0.95)
    section_pill_w = 0.18
    _pill(ax, 0.07, 0.611, section_pill_w, 0.034, "Route setup", face="#DBEAFE", text_color="#1D4ED8")
    teacher_panel = _rounded_panel(ax, 0.05, 0.19, 0.57, 0.21, face="white", edge=edge, radius=0.026, linewidth=0.95)
    _pill(ax, 0.07, 0.361, section_pill_w, 0.034, "Teacher + filter", face="#DBEAFE", text_color="#1D4ED8")
    training_panel = _rounded_panel(ax, 0.66, 0.19, 0.25, 0.46, face="white", edge=edge, radius=0.026, linewidth=0.95)
    _pill(ax, 0.685, 0.611, 0.18, 0.034, "Training variants", face="#DBEAFE", text_color="#1D4ED8")

    top_y = 0.468
    bottom_y = 0.218
    card_w = 0.16
    card_h = 0.13
    xs = [0.07, 0.255, 0.44]

    _card(xs[0], top_y, card_w, card_h, "Trusted starts", "2,177 GPS frames", "#EAF8EF", "#4ADE80", "1")
    _card(xs[1], top_y, card_w, card_h, "POI sampler", "sample up to 3 POIs", "#FFF7DB", "#F59E0B", "2")
    _card(xs[2], top_y, card_w, card_h, "OSM route", "route steps + bearings", "#EDF4FF", "#60A5FA", "3")
    _card(xs[0], bottom_y, card_w, card_h, "Teacher VLM", "6,510 raw answers", "#F3ECFF", "#A78BFA", "4")
    _card(xs[1], bottom_y, card_w, card_h, "Verifier", "6,249 valid tuples", "#FEEDEE", "#FB7185", "5")
    _card(xs[2], bottom_y, card_w, card_h, "Strict set", "4,689 kept (<30 deg)", "#EAF8EF", "#4ADE80", "6")

    _arrow((xs[0] + card_w, top_y + card_h / 2), (xs[1], top_y + card_h / 2))
    _arrow((xs[1] + card_w, top_y + card_h / 2), (xs[2], top_y + card_h / 2))
    _arrow((xs[1] + card_w / 2, top_y), (xs[1] + card_w / 2, bottom_y + card_h), rad=0.0)
    _arrow((xs[0] + card_w, bottom_y + card_h / 2), (xs[1], bottom_y + card_h / 2))
    _arrow((xs[1] + card_w, bottom_y + card_h / 2), (xs[2], bottom_y + card_h / 2))
    big_arrow_y = 0.41
    ax.add_patch(
        FancyArrowPatch(
            (0.60, big_arrow_y),
            (0.66, big_arrow_y),
            arrowstyle="-|>",
            mutation_scale=24,
            linewidth=2.6,
            color=line_color,
            connectionstyle="arc3,rad=0",
            shrinkA=2,
            shrinkB=2,
        )
    )

    train_note = ax.text(
        0.785,
        0.515,
        "Shared strict pool for all variants.",
        ha="center",
        va="center",
        fontsize=8.7,
        color=body_color,
        linespacing=1.16,
        clip_on=True,
    )
    train_note.set_clip_path(training_panel)

    styles_box = _rounded_panel(ax, 0.69, 0.31, 0.09, 0.13, face="#F8FAFF", edge=edge, radius=0.022, linewidth=0.90)
    _pill(ax, 0.703, 0.405, 0.064, 0.028, "Prompt styles", face="white", text_color=title_color, edge=edge, fontsize=6.9)
    styles_text = ax.text(0.735, 0.355, "given\nhidden\ninfer", ha="center", va="center", fontsize=8.4, color=body_color, linespacing=1.12, clip_on=True)
    styles_text.set_clip_path(styles_box)

    lora_box = _rounded_panel(ax, 0.80, 0.31, 0.09, 0.13, face="#EDF4FF", edge=edge, radius=0.022, linewidth=0.90)
    _pill(ax, 0.813, 0.405, 0.064, 0.028, "LoRA train", face="white", text_color=title_color, edge=edge, fontsize=6.9)
    lora_text = ax.text(0.845, 0.355, "student\nQwen2.5-VL-7B", ha="center", va="center", fontsize=8.0, color=body_color, linespacing=1.12, clip_on=True)
    lora_text.set_clip_path(lora_box)

    _arrow((0.78, 0.375), (0.80, 0.375))

    footer = _rounded_panel(ax, 0.19, 0.09, 0.62, 0.046, face="#E8F0FF", edge=edge, radius=0.022, linewidth=0.95)
    footer_text = ax.text(
        0.50,
        0.112,
        "Routes are synthesized first; teacher answers are filtered next; LoRA training happens on the shared strict pool.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=title_color,
        clip_on=True,
    )
    footer_text.set_clip_path(footer)
    save(fig, "instruction_pipeline.png")


def plot_pass_strict(summaries: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    values = [summaries[c.tag]["pass_strict_30"] * 100.0 for c in CONDITIONS]
    labels = [c.label for c in CONDITIONS]
    bars = ax.bar(labels, values, color=[c.color for c in CONDITIONS], edgecolor="#334155", linewidth=0.8)
    ax.axvline(2.5, color="#94A3B8", linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("PASS_strict (%)")
    ax.set_title("Overall pass rate")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=9, weight="bold")
    save(fig, "plot_pass_strict.png")


def plot_gate_pass_rates(summaries: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    matrix = np.array([[summaries[c.tag]["gate_pass_rate"][gate] * 100.0 for gate, _ in GATES] for c in CONDITIONS])
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(GATES)), [label for _, label in GATES])
    ax.set_yticks(np.arange(len(CONDITIONS)), [c.short for c in CONDITIONS])
    ax.set_title("Verifier check pass rates")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color="white" if value > 55 else "#0F172A", fontsize=9, weight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("pass rate (%)")
    save(fig, "plot_gate_pass_rates.png")


def plot_closed_loop(summaries: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), layout="constrained")
    values = [summaries[c.tag]["gate_pass_rate"]["3_closed_loop"] * 100.0 for c in CONDITIONS]
    labels = [c.label for c in CONDITIONS]
    bars = ax.bar(labels, values, color=[c.color for c in CONDITIONS], edgecolor="#334155", linewidth=0.8)
    ax.axhline(25.0, color="#64748B", linestyle="--", linewidth=1.4, label="random 4-action baseline")
    ax.axvline(2.5, color="#94A3B8", linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Closed-loop pass rate (%)")
    ax.set_title("Geometric correctness (closed-loop)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=9, weight="bold")
    save(fig, "plot_closed_loop.png")


def plot_delta_distribution() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    samples = [delta_samples(c.tag) for c in CONDITIONS]
    parts = ax.violinplot(samples, positions=np.arange(1, len(CONDITIONS) + 1), widths=0.8, showmeans=False, showmedians=True)
    for body, cond in zip(parts["bodies"], CONDITIONS):
        body.set_facecolor(cond.color)
        body.set_edgecolor("#334155")
        body.set_alpha(0.85)
    parts["cmedians"].set_color("#0F172A")
    parts["cmedians"].set_linewidth(2.0)
    ax.axhline(30.0, color="#16A34A", linestyle="--", linewidth=1.2, label="strict threshold")
    ax.axhline(55.0, color="#F59E0B", linestyle="--", linewidth=1.2, label="loose threshold")
    ax.set_xticks(np.arange(1, len(CONDITIONS) + 1), [c.short for c in CONDITIONS])
    ax.set_ylabel(r"$|\delta|$ (deg)")
    ax.set_ylim(0, 180)
    ax.set_title(r"Closed-loop $\delta$ distribution")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    save(fig, "plot_delta_distribution.png")


def plot_heading_error() -> None:
    errors = parsed_heading_errors()
    fig, ax = plt.subplots(figsize=(7.0, 4.4), layout="constrained")
    bins = np.arange(0, 181, 15)
    ax.hist(errors, bins=bins, color="#60A5FA", edgecolor="#1E3A8A", alpha=0.85)
    median = float(np.median(errors))
    p90 = percentile(errors, 0.9)
    ax.axvline(median, color="#16A34A", linestyle="--", linewidth=2.0, label=f"median = {median:.1f}")
    ax.axvline(p90, color="#DC2626", linestyle="--", linewidth=2.0, label=f"p90 = {p90:.1f}")
    ax.set_xlim(0, 180)
    ax.set_xlabel(r"$|h_{inf} - h_{gt}|$ (deg)")
    ax.set_ylabel("Number of samples")
    ax.set_title("Heading inference error (C3)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "plot_heading_error.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries()
    draw_teaser()
    draw_gps_pipeline()
    draw_instruction_pipeline()
    plot_pass_strict(summaries)
    plot_gate_pass_rates(summaries)
    plot_delta_distribution()
    plot_closed_loop(summaries)
    plot_heading_error()


if __name__ == "__main__":
    main()
