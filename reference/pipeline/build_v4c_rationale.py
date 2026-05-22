"""Build v4c training data: Claude generates the STEP 4 'infer heading' rationale.

For every v3 training sample we already know the ground-truth heading.
We ask Claude (Sonnet 4.6) to look at the image plus the nearby-POI list
and write 3-5 sentences explaining how the heading number could be
derived from visible cues. We splice that rationale into the v4b
template so the student model sees grounded, varied reasoning rather
than a templated string.

Usage
-----
    # smoke test on 5 samples first
    python pipeline/build_v4c_rationale.py --n 5

    # full run (~$30, ~1-2h with parallel workers)
    python pipeline/build_v4c_rationale.py --workers 8

Output
------
    data/cities/zurich/synth_v4c_train.jsonl
    data/cities/zurich/synth_v4c_eval.jsonl
"""

import argparse
import base64
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cities" / "zurich"

# Read API key from env or known secrets file
if "ANTHROPIC_API_KEY" not in os.environ:
    secrets = Path("/root/.secrets/anthropic.env")
    if secrets.exists():
        for ln in secrets.read_text().splitlines():
            m = re.match(r"export\s+(\w+)\s*=\s*['\"]?([^'\"]+)['\"]?\s*$", ln)
            if m:
                os.environ[m.group(1)] = m.group(2)

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
assert API_KEY, "ANTHROPIC_API_KEY missing"

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"   # strong vision, ~$3/$15 per 1M


# ───────────────────── Claude prompt ─────────────────────────────────

RATIONALE_SYSTEM = (
    "You write short reasoning chains (3-4 sentences, ~80 words max) "
    "that explain how to infer compass heading from a photograph and a "
    "nearby-POI list. Structure your paragraph as: "
    "(1) what's visible in the photo (1-2 specific objects), "
    "(2) which named POIs match those objects' positions, "
    "(3) how those POI bearings constrain the camera direction, "
    "(4) the resulting heading. "
    "Use POI names, NOT their full GPS coordinates. "
    "End your paragraph on its own line with: "
    "'INFERRED_HEADING: <integer>'. Do not preface with 'STEP 4' or any "
    "other label."
)


def rationale_user_prompt(nearby_pois_block, gt_heading_int, dest_name):
    return (
        f"The person in this photograph is walking in Zurich's old town. "
        f"Their actual compass heading is {gt_heading_int}° (0=N, 90=E, "
        f"180=S, 270=W).\n\n"
        f"Nearby POIs:\n{nearby_pois_block}\n\n"
        f"Write 3-4 SHORT sentences (≤80 words total) explaining how a "
        f"person could derive heading = {gt_heading_int}° step by step:\n"
        f"  - sentence 1: identify 1-2 visible objects in the photo\n"
        f"  - sentence 2: link them to named POIs from the list\n"
        f"  - sentence 3: reason about which direction those POIs sit "
        f"relative to the user, and what camera orientation that implies\n"
        f"  - sentence 4 (optional): conclude with the heading value\n\n"
        f"Use POI names, NOT full GPS coordinates. End with:\n\n"
        f"INFERRED_HEADING: {gt_heading_int}"
    )


def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def call_claude(image_path, nearby_pois_block, gt_heading_int, dest_name,
                retries=3, max_tokens=220):
    """Returns the rationale string, or None on failure."""
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": RATIONALE_SYSTEM,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": img_to_b64(image_path),
                }},
                {"type": "text", "text": rationale_user_prompt(
                    nearby_pois_block, gt_heading_int, dest_name)},
            ],
        }],
    }
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=60)
            if r.status_code == 200:
                data = r.json()
                txt = data["content"][0]["text"].strip()
                # sanity: must end with INFERRED_HEADING
                if "INFERRED_HEADING" not in txt.upper():
                    txt = txt + f"\nINFERRED_HEADING: {gt_heading_int}"
                return txt
            if r.status_code in (429, 503):
                wait = 2 ** attempt * 5
                time.sleep(wait)
                continue
            print(f"  HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"  err: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            time.sleep(5)
    return None


# ───────────────────── data transformation ─────────────────────────────

RX_NEARBY_BLOCK = re.compile(
    r"Nearby POIs[^\n]*:\n((?:\s*-\s*[^\n]+\n?)+)", re.I)
RX_THINKING = re.compile(r"<thinking>(.*?)</thinking>", re.S | re.I)
RX_STEP = re.compile(r"\bSTEP\s+(\d+)[^\n]*?:\s*", re.I)
RX_HEADING_PHRASE = re.compile(
    r"(?:\s+(?:and|with)\s+)?\s*,?\s*Heading\s*[:\s]\s*(?:is\s+)?\d+\.?\d*\s*°?",
    re.I)
RX_HEADING_LABEL_LINE = re.compile(r"^\s*Heading\s*:\s*\d+\.?\d*\s*°?\s*$\n?",
                                    re.M | re.I)
RX_HEADING_LINE = re.compile(r"^Your camera is facing.*\n", re.M)


def split_steps(text):
    parts = RX_STEP.split(text)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        try:
            out[int(parts[i])] = parts[i + 1].strip()
        except (ValueError, IndexError):
            continue
    return out


# ── system prompt for v4c (= v4b but mentions rationale-style STEP 4) ──

SYSTEM_PROMPT_V4C = """You are a walking-direction assistant for travelers who have trouble reading map apps. The user sends you (a) a photo from their phone camera, (b) their current GPS, (c) a list of nearby POIs with absolute coordinates, (d) an OSM-planned walking route described by the absolute bearing of its first segment, and (e) a natural-language question.

You DO NOT receive the camera's compass heading. You must infer it by combining what you see in the photograph with the absolute positions of the nearby POIs.

Output two parts in order: <thinking>...</thinking> then <answer>...</answer>. The <thinking> block has seven steps. Two structured fields must each appear on their own line:

  INFERRED_HEADING: <integer 0-359>
  FIRST_ACTION: <one of: continue ahead | turn left | turn right | turn around>

  STEP 1 (understand the question): one-sentence restatement.

  STEP 2 (resolve coordinates): user GPS, destination GPS, distance, minutes.

  STEP 3 (look at the image): list 3-5 visible things, marked LEFT / CENTER / RIGHT.

  STEP 4 (infer heading from image + map): write a short paragraph that explicitly references specific visible objects from STEP 3 and their corresponding POI positions on the map. Reason about which directions those POIs sit in (using their absolute coordinates relative to the user's GPS). Combine those bearings with which side of the frame the objects appear on, to deduce the camera's heading. End with:
      INFERRED_HEADING: <integer 0-359>

  STEP 5 (compute action):
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


def build_v4c_row(row, rationale):
    """Splice Claude's rationale into the v4b-style template."""
    out = json.loads(json.dumps(row))
    meta = out["_meta"]
    heading_gt = meta.get("user_heading")
    if heading_gt is None or rationale is None:
        return None
    heading_int = round(heading_gt) % 360
    bearing = round(meta.get("first_seg_bearing", 0))

    # 1. user_msg: drop heading line
    for m in out["messages"]:
        if m["role"] == "user":
            m["content"] = RX_HEADING_LINE.sub("", m["content"])

    # 2. system: v4c
    out["messages"][0]["content"] = SYSTEM_PROMPT_V4C

    # 3. assistant thinking: build new structure
    asst = out["messages"][2]["content"]
    th_match = RX_THINKING.search(asst)
    if not th_match:
        return None
    steps = split_steps(th_match.group(1))
    if 1 not in steps or 2 not in steps:
        return None
    # scrub heading from STEP 2
    s2 = RX_HEADING_LABEL_LINE.sub("", steps[2])
    s2 = RX_HEADING_PHRASE.sub("", s2)
    steps[2] = s2

    # action verb
    action = meta.get("first_action", "continue ahead")
    diff = ((bearing - heading_int + 540) % 360) - 180

    new_lines = [f"STEP 1: {steps.get(1, '').strip()}"]
    new_lines.append(f"STEP 2: {steps.get(2, '').strip()}")
    if 4 in steps:
        new_lines.append(f"STEP 3 (look at the image): {steps[4].strip()}")
    new_lines.append(f"STEP 4 (infer heading from image + map): {rationale.strip()}")
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

    out["messages"][2]["content"] = re.sub(
        r"<thinking>.*?</thinking>",
        f"<thinking>\n{new_thinking}\n</thinking>",
        asst, count=1, flags=re.S,
    )
    return out


def process_one(row):
    meta = row["_meta"]
    heading_gt = meta.get("user_heading")
    if heading_gt is None:
        return None
    image_path = row["image"]

    # Find the nearby POI block in user_msg (raw, before heading is stripped)
    user_msg = next(m["content"] for m in row["messages"] if m["role"] == "user")
    m = RX_NEARBY_BLOCK.search(user_msg)
    nearby_block = m.group(1).strip() if m else "(no POIs)"

    rationale = call_claude(
        image_path, nearby_block, round(heading_gt), meta.get("destination", ""))
    if rationale is None:
        return None
    return build_v4c_row(row, rationale)


def transform_jsonl(src, dst, n_max=None, workers=8):
    rows = [json.loads(l) for l in open(src) if l.strip()]
    if n_max:
        rows = rows[:n_max]
    print(f"  {src.name}: {len(rows)} samples → {dst.name} (workers={workers})")
    n_ok = n_fail = 0
    t0 = time.time()
    with open(dst, "w") as fout, \
         concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, r) for r in rows]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            try:
                row = fut.result()
            except Exception as e:
                row = None
                print(f"  err: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
            if row is None:
                n_fail += 1
            else:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_ok += 1
            if (i + 1) % 50 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                eta_min = (len(rows) - i - 1) / max(rate, 1e-3) / 60
                print(f"    [{i+1}/{len(rows)}] ok={n_ok} fail={n_fail} "
                      f"({rate:.1f}/s, ETA {eta_min:.0f} min)")
    print(f"  done: {n_ok} ok, {n_fail} failed in {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="cap to first N samples (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-eval", action="store_true",
                    help="only build the eval set (cheap test)")
    args = ap.parse_args()

    if not args.only_eval:
        transform_jsonl(DATA / "synth_v3_train.jsonl",
                         DATA / "synth_v4c_train.jsonl",
                         n_max=args.n, workers=args.workers)
    transform_jsonl(DATA / "synth_v3_eval.jsonl",
                     DATA / "synth_v4c_eval.jsonl",
                     n_max=args.n, workers=args.workers)


if __name__ == "__main__":
    main()
