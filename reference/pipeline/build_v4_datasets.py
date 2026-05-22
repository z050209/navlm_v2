"""Build v4a (implicit) and v4b (explicit) training data from v3 strict.

v4a: drop heading from user_msg + drop STEP 3 (heading math) from thinking +
     renumber. Model learns input→answer without explicit heading reasoning.

v4b: drop heading from user_msg + restructure thinking to include
     'INFERRED_HEADING: <gt>' step. Model learns to output heading from
     visual cues, then use it for the math.

Usage:
    python pipeline/build_v4_datasets.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cities" / "zurich"


# ────────────────────────────────────────────────────────────────────
# system prompts
# ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_V4A = """You are a walking-direction assistant for travelers who have trouble reading map apps. The user sends you (a) a photo from their phone camera, (b) their current GPS, (c) a list of nearby POIs with absolute coordinates, (d) an OSM-planned walking route described by the absolute bearing of its first segment, and (e) a natural-language question.

You DO NOT receive the camera's compass heading. Use the photograph and the nearby-POI map to determine which way the user is facing implicitly — your final answer must use the correct relative-direction verb (continue ahead / turn left / turn right / turn around).

Output two parts in order: <thinking>...</thinking> then <answer>...</answer>. The <thinking> block has five steps:

  STEP 1 (understand the question): one-sentence restatement, identify the destination.

  STEP 2 (resolve coordinates): user GPS, destination GPS, total walking distance and minutes.

  STEP 3 (look at the image): list 3-5 visible things in the photograph. Mark each as LEFT / CENTER / RIGHT.

  STEP 4 (pick an anchor): from STEP 3, choose ONE distinctive object the user can identify.

  STEP 5 (plan the answer): write the action verb you'll use and how the answer will reference the anchor.

The <answer> block follows the same TTS-friendly rules as v3 (2-4 short sentences, no compass words, no metres / km, contractions ok, descriptors with comma + 'that's the X')."""


SYSTEM_PROMPT_V4B = """You are a walking-direction assistant for travelers who have trouble reading map apps. The user sends you (a) a photo from their phone camera, (b) their current GPS, (c) a list of nearby POIs with absolute coordinates, (d) an OSM-planned walking route described by the absolute bearing of its first segment, and (e) a natural-language question.

You DO NOT receive the camera's compass heading. You must infer the heading from the image and the nearby-POI map, then translate the route's absolute bearing into a relative direction.

Output two parts in order: <thinking>...</thinking> then <answer>...</answer>. The <thinking> block has seven steps. Two structured fields must each appear on their own line in the exact format shown — they are parsed mechanically:

  INFERRED_HEADING: <integer 0-359>
  FIRST_ACTION: <one of: continue ahead | turn left | turn right | turn around>

  STEP 1 (understand the question): one-sentence restatement.

  STEP 2 (resolve coordinates): user GPS, destination GPS, distance, minutes.

  STEP 3 (look at the image): list 3-5 visible things, marked LEFT / CENTER / RIGHT.

  STEP 4 (infer heading from image): combine visible objects in STEP 3 with the nearby-POI map to deduce which way the camera is facing. End with:
      INFERRED_HEADING: <integer 0-359>

  STEP 5 (compute action): show this arithmetic explicitly:
       diff = (route_bearing - INFERRED_HEADING + 540) mod 360 - 180
       if |diff| <= 35°       -> continue ahead
       elif |diff| > 135°     -> turn around
       elif diff < 0          -> turn left
       else                   -> turn right
    End with:
      FIRST_ACTION: <verb>

  STEP 6 (pick an anchor): one distinctive visible object to anchor the answer to.

  STEP 7 (plan the answer): brief plan referencing action verb + anchor.

The <answer> block follows the same TTS-friendly rules (2-4 short sentences, no compass / metres, descriptors with comma + 'that's the X')."""


# ────────────────────────────────────────────────────────────────────
# transformers
# ────────────────────────────────────────────────────────────────────

RX_HEADING_LINE = re.compile(r"^Your camera is facing.*\n", re.M)
# Catches all variants of compass heading + a number:
#   "heading 243°", "Heading: 243°", "heading is 243",
#   ", heading 243", "and heading 243", "with heading 243"
# Bounded by  a number to avoid eating "heading toward X" (idiomatic).
RX_HEADING_PHRASE = re.compile(
    r"(?:\s+(?:and|with)\s+)?\s*,?\s*Heading\s*[:\s]\s*(?:is\s+)?\d+\.?\d*\s*°?",
    re.I)
# Also catches a standalone "Heading: 243°" line at start of a line
RX_HEADING_LABEL_LINE = re.compile(r"^\s*Heading\s*:\s*\d+\.?\d*\s*°?\s*$\n?",
                                    re.M | re.I)
RX_THINKING = re.compile(r"<thinking>(.*?)</thinking>", re.S | re.I)
RX_STEP = re.compile(r"\bSTEP\s+(\d+)[^\n]*?:\s*", re.I)


def split_steps(thinking_text):
    """Split v3 thinking into {step_num: content} dict."""
    parts = RX_STEP.split(thinking_text)
    # parts: [pre, '1', body1, '2', body2, ...]
    out = {}
    for i in range(1, len(parts) - 1, 2):
        try:
            n = int(parts[i])
            out[n] = parts[i + 1].strip()
        except (ValueError, IndexError):
            continue
    return out


def make_v4a(row):
    """v3 → v4a: drop heading mentions; remove STEP 3 (math); renumber."""
    out = json.loads(json.dumps(row))  # deep copy

    # 1. user_msg: drop heading line
    for m in out["messages"]:
        if m["role"] == "user":
            m["content"] = RX_HEADING_LINE.sub("", m["content"])

    # 2. system: replace with v4a
    out["messages"][0]["content"] = SYSTEM_PROMPT_V4A

    # 3. assistant thinking: drop STEP 3, scrub heading words from STEP 2
    asst = out["messages"][2]["content"]
    th_match = RX_THINKING.search(asst)
    if not th_match:
        return None
    steps = split_steps(th_match.group(1))
    if 1 not in steps or 2 not in steps:
        return None
    if 2 in steps:
        s2 = RX_HEADING_LABEL_LINE.sub("", steps[2])
        s2 = RX_HEADING_PHRASE.sub("", s2)
        steps[2] = s2
    # v3: 1=question, 2=resolve, 3=math, 4=visible, 5=anchor, 6=plan
    # v4a renumber: 1, 2, (skip 3), 4→3, 5→4, 6→5
    new_lines = [f"STEP 1: {steps.get(1, '').strip()}"]
    new_lines.append(f"STEP 2: {steps.get(2, '').strip()}")
    if 4 in steps:
        new_lines.append(f"STEP 3 (look at the image): {steps[4].strip()}")
    if 5 in steps:
        new_lines.append(f"STEP 4 (pick an anchor): {steps[5].strip()}")
    if 6 in steps:
        new_lines.append(f"STEP 5 (plan the answer): {steps[6].strip()}")
    new_thinking = "\n\n".join(new_lines)
    new_asst = re.sub(
        r"<thinking>.*?</thinking>",
        f"<thinking>\n{new_thinking}\n</thinking>",
        asst, count=1, flags=re.S,
    )
    out["messages"][2]["content"] = new_asst
    return out


def make_v4b(row):
    """v3 → v4b: drop heading from input; add INFERRED_HEADING step using gt."""
    out = json.loads(json.dumps(row))

    # 1. user_msg: drop heading line
    for m in out["messages"]:
        if m["role"] == "user":
            m["content"] = RX_HEADING_LINE.sub("", m["content"])

    # 2. system: replace with v4b
    out["messages"][0]["content"] = SYSTEM_PROMPT_V4B

    # 3. assistant thinking: insert INFERRED_HEADING step using gt
    meta = out["_meta"]
    heading_gt = meta.get("user_heading")
    if heading_gt is None:
        return None
    heading_int = round(heading_gt) % 360
    bearing = round(meta.get("first_seg_bearing", 0))

    asst = out["messages"][2]["content"]
    th_match = RX_THINKING.search(asst)
    if not th_match:
        return None
    steps = split_steps(th_match.group(1))
    if 1 not in steps or 2 not in steps:
        return None
    if 2 in steps:
        s2 = RX_HEADING_LABEL_LINE.sub("", steps[2])
        s2 = RX_HEADING_PHRASE.sub("", s2)
        steps[2] = s2

    # action verb (from existing meta — guaranteed correct)
    action = meta.get("first_action", "continue ahead")
    diff = ((bearing - heading_int + 540) % 360) - 180

    # v4b ordering:
    #   1=question, 2=resolve(no heading),
    #   3=visible(=v3 STEP 4),
    #   4=infer heading + INFERRED_HEADING,
    #   5=math + FIRST_ACTION (using v3 STEP 3 math but referencing INFERRED_HEADING),
    #   6=anchor (=v3 STEP 5),
    #   7=plan (=v3 STEP 6).
    new_lines = [f"STEP 1: {steps.get(1, '').strip()}"]
    new_lines.append(f"STEP 2: {steps.get(2, '').strip()}")
    if 4 in steps:
        new_lines.append(f"STEP 3 (look at the image): {steps[4].strip()}")
    new_lines.append(
        "STEP 4 (infer heading from image): Looking at the visible scene and "
        "matching the surrounding context to the nearby-POI map, I infer "
        f"that I am facing approximately {heading_int}°.\n"
        f"INFERRED_HEADING: {heading_int}"
    )
    new_lines.append(
        f"STEP 5 (compute action):\n"
        f"diff = ({bearing} - {heading_int} + 540) mod 360 - 180 = {diff}°\n"
        + (f"|{abs(diff)}| <= 35° → continue ahead\n" if abs(diff) <= 35 else
           f"|{abs(diff)}| > 135° → turn around\n" if abs(diff) > 135 else
           f"35 < |{abs(diff)}| <= 135 and diff {'<' if diff < 0 else '>='} 0 → "
           f"{'turn left' if diff < 0 else 'turn right'}\n")
        + f"FIRST_ACTION: {action}"
    )
    if 5 in steps:
        new_lines.append(f"STEP 6 (pick an anchor): {steps[5].strip()}")
    if 6 in steps:
        new_lines.append(f"STEP 7 (plan the answer): {steps[6].strip()}")
    new_thinking = "\n\n".join(new_lines)
    new_asst = re.sub(
        r"<thinking>.*?</thinking>",
        f"<thinking>\n{new_thinking}\n</thinking>",
        asst, count=1, flags=re.S,
    )
    out["messages"][2]["content"] = new_asst
    return out


def transform_jsonl(src, dst, fn, label):
    n_in = n_out = 0
    with open(src) as fin, open(dst, "w") as fout:
        for ln in fin:
            n_in += 1
            row = json.loads(ln)
            new = fn(row)
            if new is None:
                continue
            fout.write(json.dumps(new, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"  {label:<10}  {src.name:<40}  {n_in:>5} → {n_out:>5}")


def main():
    pairs = [
        ("synth_v3_train.jsonl", "synth_v4a_train.jsonl", make_v4a, "v4a-train"),
        ("synth_v3_eval.jsonl",  "synth_v4a_eval.jsonl",  make_v4a, "v4a-eval"),
        ("synth_v3_train.jsonl", "synth_v4b_train.jsonl", make_v4b, "v4b-train"),
        ("synth_v3_eval.jsonl",  "synth_v4b_eval.jsonl",  make_v4b, "v4b-eval"),
    ]
    print(f"{'tag':<12}  {'src':<40}  {'in':>5} → {'out':>5}")
    print("-" * 72)
    for src_name, dst_name, fn, label in pairs:
        transform_jsonl(DATA / src_name, DATA / dst_name, fn, label)


if __name__ == "__main__":
    main()
