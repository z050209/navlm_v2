"""Output parsing + verification.

Two layers:
  1. `parse_assistant` + `verify`: cheap, format-only checks on the teacher
     model's output (regex-level — no extra VLM call).
  2. `visual_verify_poi`: directed Gemma yes/no question — "is the SCENE
     consistent with this POI?" — used to gate frames before generation.
"""

import re
import sys
from pathlib import Path

import requests

# Allow imports either as a package or via sys.path
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from synth_utils import has_forbidden_terms, img_to_data_url  # noqa: E402

from .prompts import VERIFIER_PROMPT_TEMPLATE


# ---------- output parsing ----------

_THINK_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.S)
_ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def parse_assistant(text):
    """Pull the <thinking> and <answer> blocks out of teacher output."""
    t = _THINK_RE.search(text or "")
    a = _ANS_RE.search(text or "")
    return (t.group(1).strip() if t else None,
            a.group(1).strip() if a else None)


def verify(thinking, answer):
    """Light format checks for the v2 prompt — just makes sure both blocks
    are present and the answer doesn't violate TTS / scene-anchor rules.

    Heavy verification (heading, action, closed-loop) lives in
    `pipeline/step_12_closed_loop_verify.py`, which is run separately
    after generation.
    """
    fails = []
    if thinking is None:
        fails.append("missing_thinking")
    if answer is None:
        fails.append("missing_answer")
    if answer:
        forb = has_forbidden_terms(answer)
        if forb:
            fails.append(f"forbidden_in_answer:{forb}")
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
        if not (2 <= len(sents) <= 4):
            fails.append(f"sentence_count:{len(sents)}")
    # NB: forbidden terms in <thinking> are intentionally allowed —
    # the v2 CoT must reason in compass words, GPS coords, and degree
    # numbers. Step 12 verifier handles full structural checks.
    return fails


# ---------- directed visual verifier ----------

def visual_verify_poi(image_path, poi_name, vllm_url, model, timeout=45):
    """Directed visual check: 'is the SCENE consistent with <POI>?'.

    Designed to catch reverse-lookup mistakes — if we claim user is at
    Hauptbahnhof but the image shows Landesmuseum, this should reject.

    Importantly this is a SCENE-match question (architecture, neighbouring
    landmarks), NOT a literal-text-visibility check. Earlier versions
    misfired because Gemma read 'Is X visible' as 'does the word X appear'.

    Returns
    -------
    (passed: bool, raw_response: str, parsed: 'yes'|'no'|'unsure')
    """
    prompt = VERIFIER_PROMPT_TEMPLATE.format(poi=poi_name)
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": img_to_data_url(Path(image_path))}},
            ]}],
            "max_tokens": 60,
            "temperature": 0.0,
        }
        r = requests.post(f"{vllm_url}/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # Don't block on verifier errors; mark as unsure so the sample
        # passes through downstream filters.
        return True, f"verifier_error:{type(e).__name__}", "unsure"

    head = out.lstrip().upper()
    if head.startswith("YES"):
        return True, out, "yes"
    if head.startswith("NO"):
        return False, out, "no"
    if head.startswith("UNSURE"):
        return True, out, "unsure"  # accept ambiguous, don't over-filter
    if "yes" in out.lower()[:20]:
        return True, out, "yes"
    if "no" in out.lower()[:5]:
        return False, out, "no"
    return True, out, "unsure"
