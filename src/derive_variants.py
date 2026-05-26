"""Derive the three training views from one annotation file.

  given      — keep "My camera heading: ..." in the user msg, and the
                INFERRED_HEADING step in <thinking>. (with-compass
                LoRA — L-given)
  implicit   — strip the heading line from the user msg AND drop the
                INFERRED_HEADING step from <thinking>. (no heading
                anywhere — L-implicit LoRA)
  explicit   — strip the heading line from the user msg BUT KEEP the
                INFERRED_HEADING step in <thinking>. (model must learn
                to *infer* the heading from the photo — L-explicit LoRA)

Input:  data/cities/zurich/eval_train.jsonl   (rows from src.eval_split)
Output: data/sft/{given,implicit,explicit}.jsonl

Each output row is the **Qwen2.5-VL chat-template** shape that
`train_modal.py` consumes:

  {
    "image_rel": "<video>/<frame_id>.jpg",     # resolved against a
                                                # variant-aware root
    "messages": [
      {"role": "system",
       "content": [{"type": "text", "text": "<system prompt>"}]},
      {"role": "user",
       "content": [{"type": "image"},          # placeholder
                   {"type": "text", "text": "<user msg>"}]},
      {"role": "assistant",
       "content": [{"type": "text",
                    "text": "<thinking>...</thinking><answer>...</answer>"}]},
    ],
  }

The `{"type": "image"}` placeholder is what `processor.apply_chat_template`
needs in order to splice the right number of vision-token IDs into the
text — without it the model trains text-only and never attends to the
image.

  python -m src.derive_variants
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                       # noqa: E402

HEADING_LINE = re.compile(r"^My camera heading:.*\n?", re.M)
HEADING_STEP = re.compile(
    r"^(?:STEP\s*\d+\s+)?INFERRED_HEADING:.*\n?", re.M | re.I)
HEADING_REASONING = re.compile(
    r"^HEADING_REASONING:.*?(?=^[A-Z_]{3,}:|\Z)", re.M | re.S | re.I)


SYS_PROMPTS = {
    "given": (
        "You are a Zurich-local walking-tour guide. You give SPOKEN "
        "directions to a tourist who has sent you a phone photo of "
        "what they currently see plus their GPS, camera heading, and "
        "destination. Reply with <thinking>...</thinking>"
        "<answer>...</answer>; the answer is 2-4 short sentences, no "
        "compass words, no numbers, no GPS, anchored to a visible "
        "object."),
    "implicit": (
        "You are a Zurich-local walking-tour guide. You give SPOKEN "
        "directions to a tourist who has sent you a phone photo of "
        "what they currently see plus their GPS and destination. They "
        "have NO compass. Reply with <thinking>...</thinking>"
        "<answer>...</answer>; the answer is 2-4 short sentences, no "
        "compass words, no numbers, no GPS, anchored to a visible "
        "object."),
    "explicit": (
        "You are a Zurich-local walking-tour guide. You give SPOKEN "
        "directions to a tourist who has sent you a phone photo of "
        "what they currently see plus their GPS and destination. They "
        "have NO compass — INFER their facing from the photo. Reply "
        "with <thinking>...</thinking><answer>...</answer>; INSIDE "
        "<thinking> include a line 'INFERRED_HEADING: <0-359>' where "
        "you commit to your best guess of which way the camera is "
        "facing, reasoning from visible landmarks. The answer is 2-4 "
        "short sentences, no compass words, no numbers, no GPS, "
        "anchored to a visible object."),
}


def strip_heading_from_user(msg: str) -> str:
    return HEADING_LINE.sub("", msg).strip("\n")


def strip_heading_step(thinking: str) -> str:
    t = HEADING_STEP.sub("", thinking)
    t = HEADING_REASONING.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def build_user_msg(rec: dict, variant: str) -> str:
    """Reconstruct the user message exactly the way eval_modal.py
    will at inference time. Heading line included only for 'given'."""
    lat, lon = rec["gps"]
    lines = [f"My GPS: {lat:.5f}, {lon:.5f}"]
    if variant == "given":
        lines.append(f"My camera heading: {rec['heading']:.0f}° "
                     "(0=N, 90=E)")
    lines += [
        f"Destination: {rec['dest_name']} "
        f"(first-segment bearing {rec['route_bearing']:.0f}°)",
        f"Walking-route distance: {rec['route_distance_m']:.0f} m",
    ]
    if rec.get("nearby_pois"):
        lines.append("Nearby POIs:")
        lines += [f"- {n}" for n in rec["nearby_pois"][:8]]
    lines.append("Tell me what to do, in 2-4 spoken sentences, "
                 "anchored to something I can actually see in the "
                 "photo.")
    return "\n".join(lines)


def to_message_row(rec: dict, variant: str) -> dict:
    """Build the {image_rel, messages[]} row train_modal.py consumes."""
    user_msg = build_user_msg(rec, variant)

    thinking = rec.get("thinking", "")
    answer = rec.get("answer", "")
    if variant == "implicit":
        thinking = strip_heading_step(thinking)
    assistant_text = (f"<thinking>\n{thinking}\n</thinking>\n"
                      f"<answer>\n{answer}\n</answer>")

    return {
        "image_rel": f"{rec['video']}/{rec['frame_id']}.jpg",
        "variant": variant,
        "messages": [
            {"role": "system",
             "content": [{"type": "text",
                          "text": SYS_PROMPTS[variant]}]},
            {"role": "user",
             "content": [{"type": "image"},
                         {"type": "text", "text": user_msg}]},
            {"role": "assistant",
             "content": [{"type": "text", "text": assistant_text}]},
        ],
        "_meta": {"video": rec["video"], "frame_id": rec["frame_id"],
                  "dest_name": rec["dest_name"]},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--input",
                    default=str(config.CITY_DIR / "eval_train.jsonl"))
    ap.add_argument("--output-dir",
                    default=str(config.DATA_ROOT / "sft"))
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in in_path.open(encoding="utf-8")
            if l.strip()]
    print(f"[derive_variants] training rows: {len(rows)}", flush=True)

    for variant in ("given", "implicit", "explicit"):
        out_path = out_dir / f"{variant}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_message_row(r, variant),
                                   ensure_ascii=False) + "\n")
        print(f"[derive_variants] {variant:9s} -> {out_path} "
              f"({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
