"""Step 12 — closed-loop verifier on synth_unified output (v3 verifier).

For each (image, prompt, assistant_response):

  1. parse the action verb from the natural-language <answer>
  2. closed-loop:
       δ = |angle_diff(heading_gt + ACTION_DELTA[answer_verb], first_seg_bearing)|
       PASS if δ ≤ 30°  (matching closed_loop_sanity_v2 baseline of 67.5%)
       WARN if 30° < δ ≤ 55°  (within action-discretization tolerance)
       FAIL if δ > 55°  (math broken — answer points wrong way)
  3. format checks on <answer> (TTS rules, no compass / metres / GPS)
  4. sentence count 2-4

Run:
    python -m pipeline.step_12_closed_loop_verify \\
        --in  data/cities/zurich/synth_unified.jsonl \\
        --out data/cities/zurich/synth_unified_verified.jsonl
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Late import — only when --recompute-bearing is set
_planner = None
_pois = None


ACTION_DELTA = {
    "continue ahead": 0.0,
    "turn left":      -90.0,
    "turn right":      90.0,
    "turn around":    180.0,
}


# Map common verb phrases found in natural-language answers to canonical actions.
# Order matters — longer / more specific patterns first.
_ANSWER_VERB_PATTERNS = [
    (r"\bturn\s+around\b",         "turn around"),
    (r"\bgo\s+back\b",             "turn around"),
    (r"\bdouble\s+back\b",         "turn around"),
    (r"\bturn\s+left\b",           "turn left"),
    (r"\bhead\s+left\b",           "turn left"),
    (r"\bbear\s+left\b",           "turn left"),
    (r"\btake\s+a\s+left\b",       "turn left"),
    (r"\bturn\s+right\b",          "turn right"),
    (r"\bhead\s+right\b",          "turn right"),
    (r"\bbear\s+right\b",          "turn right"),
    (r"\btake\s+a\s+right\b",      "turn right"),
    (r"\bcontinue\s+(straight\s+)?ahead\b", "continue ahead"),
    (r"\bcontinue\s+straight\b",   "continue ahead"),
    (r"\bgo\s+straight\b",         "continue ahead"),
    (r"\bwalk\s+straight\b",       "continue ahead"),
    (r"\bkeep\s+(going|walking|heading)\b", "continue ahead"),
    (r"\bwalk\s+ahead\b",          "continue ahead"),
    (r"\bwalk\s+forward\b",        "continue ahead"),
]


RX_THINKING = re.compile(r"<thinking>(.*?)</thinking>", re.S | re.I)
RX_ANSWER   = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)

RX_COMPASS  = re.compile(
    r"\b(north|south|east|west|northeast|northwest|southeast|southwest|"
    r"NE|NW|SE|SW|NNE|NNW|ENE|WNW|ESE|WSW|SSE|SSW)\b", re.I)
RX_DEGREE   = re.compile(r"\d+\s*°|\bdeg(rees?)?\b", re.I)
RX_DISTANCE = re.compile(
    r"\b\d+(\.\d+)?\s*(m|metre|meter|metres|meters|km|kilometre|kilometer|"
    r"kilometres|kilometers|block|blocks)\b", re.I)
RX_GPS      = re.compile(r"\b\d{1,3}\.\d{3,}\s*[,°]\s*\d{1,3}\.\d{3,}")


def angle_diff(a, b):
    return ((a - b + 540) % 360) - 180


def parse_assistant(text):
    th = RX_THINKING.search(text)
    an = RX_ANSWER.search(text)
    return (th.group(1).strip() if th else None,
            an.group(1).strip() if an else None)


def parse_action_from_answer(answer):
    """Find the FIRST action verb in the answer text. Returns (verb, span) or (None, None)."""
    if not answer:
        return None, None
    s = answer.lower()
    best = None
    for pat, verb in _ANSWER_VERB_PATTERNS:
        m = re.search(pat, s)
        if m and (best is None or m.start() < best[1]):
            best = (verb, m.start())
    if best:
        return best[0], best[1]
    return None, None


def gate_format(answer):
    if "—" in answer or "—" in answer:
        return False, "uses em-dash"
    if "(" in answer or ")" in answer:
        return False, "uses parenthesis"
    if "\n" in answer.strip():
        return False, "has line breaks"
    m = RX_COMPASS.search(answer)
    if m:
        return False, f"compass word: {m.group(0)!r}"
    if RX_DEGREE.search(answer):
        return False, "contains degree / 'degree'"
    m = RX_DISTANCE.search(answer)
    if m:
        return False, f"numeric distance: {m.group(0)!r}"
    if RX_GPS.search(answer):
        return False, "contains lat/lon coordinates"
    return True, None


def gate_sentence_count(answer):
    sents = [s.strip() for s in re.split(r"[.!?]+\s+", answer) if s.strip()]
    if not (2 <= len(sents) <= 4):
        return False, f"sentence count = {len(sents)}"
    for s in sents:
        n_words = len(s.split())
        if n_words > 22:
            return False, f"sentence too long ({n_words} words)"
    return True, None


RX_CHECKPOINT = re.compile(
    r"when you (?:reach|get to|see|arrive at)\s+(.+?)[,.]", re.I)


def gate_checkpoint_grounded(answer, user_msg):
    """If the answer uses a multi-turn checkpoint phrase, the named
    checkpoint must appear in the route block of the user_msg (mode B)
    OR be plausible STEP-4-derived language (mode A fallback).

    Returns (ok, reason) — True if no checkpoint phrase is used (short
    route) or the named checkpoint is grounded.
    """
    m = RX_CHECKPOINT.search(answer)
    if not m:
        return True, "no multi-turn checkpoint phrase"
    cp = m.group(1).strip().lower()
    # Drop leading articles
    cp_clean = re.sub(r"^(the|a|an|that's the|that is the)\s+", "", cp)
    # Mode B: name appears in the user_msg route block
    if cp_clean and cp_clean in user_msg.lower():
        return True, f"checkpoint matches user_msg: {cp_clean!r}"
    # Mode A fallback heuristic — common permanent-landmark words
    permanent_words = ("building", "statue", "fountain", "tower", "church",
                       "bridge", "square", "plaza", "gate", "monument",
                       "facade", "wall", "sign")
    if any(w in cp for w in permanent_words):
        return True, f"checkpoint uses permanent-landmark word: {cp!r}"
    return False, f"unverified checkpoint: {cp!r}"


def gate_closed_loop(parsed_action, heading_gt, first_seg_bearing):
    if parsed_action not in ACTION_DELTA:
        return False, "no parseable action verb in answer"
    if heading_gt is None:
        return False, "missing heading_gt"
    if first_seg_bearing is None:
        return False, "missing first_seg_bearing"
    h_walks = (heading_gt + ACTION_DELTA[parsed_action]) % 360
    delta = abs(angle_diff(h_walks, first_seg_bearing))
    if delta > 55:
        return False, f"δ={delta:.1f}° > 55° (math broken)"
    tag = "PASS<30" if delta < 30 else f"PASS_WARN<55 (δ={delta:.1f}°)"
    return True, tag


def get_first_seg_bearing(meta):
    """Use stored value if present, else recompute via planner."""
    b = meta.get("first_seg_bearing")
    if b is not None:
        return b
    # Recompute on-the-fly
    global _planner, _pois
    if _planner is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "toolbox"))
        from way_planner import WayPlanner
        _planner = WayPlanner(str(Path(__file__).resolve().parent.parent
                                   / "data/cities/zurich/osm_walking.pkl"))
    if _pois is None:
        with open(Path(__file__).resolve().parent.parent
                  / "data/cities/zurich/landmarks_zurich_osm.json") as f:
            _pois = json.load(f)
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "toolbox"))
            from scenery_pois import SCENERY_POIS
            for n, p in SCENERY_POIS.items():
                _pois.setdefault(n, {"lat": p["lat"], "lon": p["lon"]})
        except ImportError:
            pass
    dest = meta.get("destination")
    if dest not in _pois:
        return None
    p = _pois[dest]
    plan = _planner.plan(tuple(meta["user_gps"]), (p["lat"], p["lon"]),
                         user_heading=meta.get("user_heading"))
    return plan.get("first_seg_bearing") if plan else None


def verify_sample(row):
    meta = row.get("_meta", {})
    asst_msg = next((m["content"] for m in row["messages"]
                     if m["role"] == "assistant"), "")
    thinking, answer = parse_assistant(asst_msg)
    parsed_action, _pos = parse_action_from_answer(answer)

    gates = {}
    if not answer:
        gates["1_format"] = (False, "no <answer> block")
        gates["2_sentence_count"] = (False, "no <answer> block")
        gates["4_checkpoint"] = (False, "no <answer> block")
    else:
        gates["1_format"] = gate_format(answer)
        gates["2_sentence_count"] = gate_sentence_count(answer)
        user_msg = next((m["content"] for m in row["messages"]
                         if m["role"] == "user"), "")
        gates["4_checkpoint"] = gate_checkpoint_grounded(answer, user_msg)

    seg_b = get_first_seg_bearing(meta)
    gates["3_closed_loop"] = gate_closed_loop(parsed_action,
                                               meta.get("user_heading"), seg_b)

    return all(g[0] for g in gates.values()), gates, parsed_action, seg_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="data/cities/zurich/synth_unified.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    print(f"loaded {len(rows)} samples from {args.inp}")

    gate_pass = Counter()
    gate_fail_reasons = {}
    overall_pass = 0
    delta_buckets = Counter()
    fout = open(args.out, "w") if args.out else None

    for r in rows:
        ok, gates, parsed_action, seg_b = verify_sample(r)
        for name, (g_ok, reason) in gates.items():
            if g_ok:
                gate_pass[name] += 1
            else:
                gate_fail_reasons.setdefault(name, []).append(reason or "")
        if ok:
            overall_pass += 1
            if fout:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Bucket the delta even when gate passes
        if parsed_action in ACTION_DELTA:
            heading_gt = r["_meta"].get("user_heading")
            if heading_gt is not None and seg_b is not None:
                d = abs(angle_diff((heading_gt + ACTION_DELTA[parsed_action]) % 360, seg_b))
                if d < 30: delta_buckets["<30°"] += 1
                elif d <= 55: delta_buckets["30-55°"] += 1
                else: delta_buckets[">55°"] += 1
        else:
            delta_buckets["no_verb"] += 1

    if fout:
        fout.close()

    print(f"\n=== gate pass rates ===")
    for name in ["1_format", "2_sentence_count", "3_closed_loop", "4_checkpoint"]:
        n_pass = gate_pass[name]
        pct = 100 * n_pass / max(len(rows), 1)
        print(f"  {name:<20}  {n_pass:>4d}/{len(rows)}  {pct:5.1f}%")

    print(f"\n=== closed-loop δ distribution ===")
    for k in ["<30°", "30-55°", ">55°", "no_verb"]:
        n = delta_buckets[k]
        pct = 100 * n / max(len(rows), 1)
        print(f"  {k:<10}  {n:>4d}  {pct:5.1f}%")

    print(f"\n=== overall (all 3 gates) ===")
    print(f"  PASS  {overall_pass}/{len(rows)}  "
          f"({100 * overall_pass / max(len(rows), 1):.1f}%)")

    if args.out:
        print(f"\nverified rows  →  {args.out}")

    print(f"\n=== top fail reasons ===")
    for name in ["1_format", "2_sentence_count", "3_closed_loop"]:
        if name not in gate_fail_reasons or not gate_fail_reasons[name]:
            continue
        c = Counter(gate_fail_reasons[name])
        print(f"  {name}:")
        for reason, n in c.most_common(3):
            print(f"     {n:>3}× {reason[:90]}")


if __name__ == "__main__":
    main()
