"""Evaluation metrics — slide 4 of milestone2.

Four metrics, AND-ed into PASS_strict:

  (a) format_compliance      — answer format / 2-4 sentences / no compass
  (b) directional_accuracy   — closed-loop angular check, δ < 30°
  (c) checkpoint_validity    — check-in target must be a permanent route POI
  (d) anchor_faithfulness    — VLM yes/no: is the anchor object visible?

(a), (b), (c) are pure functions (unit-testable). (d) needs a Gemini
call — kept here so the API is one import away, but it is the only
metric with a side effect.

  from src.eval_metrics import (
      format_compliance, directional_accuracy, checkpoint_validity,
      anchor_faithfulness, pass_strict, parse_action_verb,
      extract_anchor, extract_checkpoint,
  )
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.routing import ACTION_DELTA, closed_loop_delta   # noqa: E402

# canonical compass words / numbers that must NOT appear in <answer>
COMPASS_FORBIDDEN = re.compile(
    r"\b(north|south|east|west|northeast|northwest|southeast|southwest|"
    r"n\.?e\.?|n\.?w\.?|s\.?e\.?|s\.?w\.?|\d{1,3}\s*°|degrees?)\b",
    re.I)
GPS_FORBIDDEN = re.compile(r"\b\d{1,2}\.\d{4,}\b")    # 47.3749 style


# ── parsing helpers ─────────────────────────────────────────────────
def parse_action_verb(answer):
    """First action verb found in `answer`, or None. Pure."""
    for verb in ACTION_DELTA.keys():
        if re.search(r"\b" + re.escape(verb) + r"\b", answer or "", re.I):
            return verb
    return None


def extract_anchor(answer):
    """Pull the anchor phrase out of an answer like
    'Turn left at the tram tracks ...' Heuristic — looks for the verb,
    then 'at|near|by|past|along|next to' followed by 1-6 words. Pure.
    `\\w` is Unicode-aware here so non-ASCII POI names ('Grossmünster',
    'Bürkliplatz') match."""
    if not answer:
        return None
    m = re.search(
        r"(?:turn\s+(?:left|right|around)|continue\s+ahead)\s+"
        r"(?:at|near|by|past|along|next to|towards?)\s+"
        r"(?:the\s+)?([^\s.,;!?][^\.,;!?]{0,60}?)\s*[.,;!?]",
        answer, re.I | re.U)
    return m.group(1).strip() if m else None


def extract_checkpoint(answer):
    """Pull the check-in target from
    '... when you reach <X>, send me another photo'. None if absent.
    Permits Unicode names ('Grossmünster')."""
    if not answer:
        return None
    m = re.search(
        r"when\s+you\s+reach\s+(?:the\s+)?([^\s.,;!?][^\.,;!?]{0,60}?)"
        r"\s*[,.]", answer, re.I | re.U)
    return m.group(1).strip() if m else None


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "")
            if s.strip()]


# ── (a) format compliance ────────────────────────────────────────────
def format_compliance(raw, answer):
    """Both blocks present, answer is 2-4 sentences, no compass/GPS.
    Returns (bool, reasons[]). Pure — unit-tested."""
    reasons = []
    if "<thinking>" not in raw or "</thinking>" not in raw:
        reasons.append("no <thinking>")
    if "<answer>" not in raw or "</answer>" not in raw:
        reasons.append("no <answer>")
    n_sent = len(_split_sentences(answer or ""))
    if not (2 <= n_sent <= 4):
        reasons.append(f"sentences={n_sent} (need 2-4)")
    if COMPASS_FORBIDDEN.search(answer or ""):
        reasons.append("compass/number leaked")
    if GPS_FORBIDDEN.search(answer or ""):
        reasons.append("GPS leaked")
    return (len(reasons) == 0), reasons


# ── (b) directional accuracy ─────────────────────────────────────────
def directional_accuracy(heading, action_verb, route_bearing,
                         max_delta=30.0):
    """Closed-loop δ < max_delta. Returns (bool, delta). Pure."""
    if action_verb is None or action_verb not in ACTION_DELTA:
        return False, None
    delta = closed_loop_delta(heading, action_verb, route_bearing)
    return delta < max_delta, delta


# ── (c) checkpoint validity ──────────────────────────────────────────
PERMANENT_KINDS = {
    "amenity=place_of_worship", "amenity=townhall", "amenity=theatre",
    "amenity=cinema", "amenity=museum", "amenity=university",
    "railway=station", "historic=monument", "historic=castle",
    "historic=memorial", "tourism=attraction", "tourism=viewpoint",
    "leisure=park", "place=square", "highway=primary",
    "highway=secondary", "highway=residential", "highway=pedestrian",
    "highway=living_street", "man_made=bridge", "waterway=river",
    "natural=water",
}
MOVABLE_WORDS = re.compile(
    r"\b(car|truck|bus|van|bike|bicycle|tram|train|person|people|"
    r"pedestrian|cyclist|dog|stroller|umbrella|sign|cone|barrier)\b",
    re.I)


def checkpoint_validity(answer, route_poi_names, all_pois_by_name,
                        route_distance_m=None):
    """A long route (>3 turns) ends with 'when you reach X'. X must be:
    (i) a real POI on the route, (ii) a permanent feature (not a
    movable object).

    Single-turn answers (no check-in needed) pass automatically.

    route_poi_names    : list[str] of OSM POIs on the planned route.
    all_pois_by_name   : dict name -> {osm_kind, ...} for permanence lookup.
    Pure — unit-tested."""
    checkpoint = extract_checkpoint(answer)
    # If no checkpoint, this metric is trivially passed (it only
    # applies to long routes).
    if checkpoint is None:
        return True, []

    reasons = []
    if MOVABLE_WORDS.search(checkpoint):
        reasons.append(f"movable: {checkpoint}")

    # case-insensitive membership in the route POIs
    canon = checkpoint.lower().strip()
    route_set = {n.lower().strip() for n in (route_poi_names or [])}
    on_route = canon in route_set or any(canon in n or n in canon
                                         for n in route_set)
    if not on_route:
        reasons.append(f"not on planned route: {checkpoint}")

    # permanence — match against OSM kind
    matched_poi = next((p for n, p in (all_pois_by_name or {}).items()
                        if n.lower().strip() == canon), None)
    if matched_poi:
        if matched_poi.get("osm_kind") not in PERMANENT_KINDS:
            reasons.append(
                f"impermanent kind: {matched_poi.get('osm_kind')}")
    return (len(reasons) == 0), reasons


# ── (d) anchor faithfulness — needs a VLM call ───────────────────────
ANCHOR_SYS = (
    "You are a careful visual checker. Given a photo and a phrase, "
    "answer YES if the named object is clearly visible in the photo, "
    "NO if it is not. Reply with a single word: YES or NO.")


def anchor_faithfulness(image_path, anchor, gemini_caller=None):
    """VLM yes/no — is `anchor` visible in the photo?

    Returns (bool, raw_response). If anchor is None, returns (True, '')
    — no anchor means nothing to hallucinate (the format check covers
    "answer had no anchor"; this metric is about *truthfulness* of the
    anchor that IS there).

    `gemini_caller`: callable (image_path, sys, user) -> str. Defaults
    to `src.gemini_api.call_gemini` with `gemini-2.5-pro`. Inject a
    fake in tests."""
    if not anchor:
        return True, ""
    if gemini_caller is None:
        from src.gemini_api import call_gemini
        import config

        def gemini_caller(img, sys_p, user_p):
            return call_gemini(img, sys_p, user_p,
                               model=config.GEMINI_GEOCHECK,
                               max_tokens=8, temperature=0.0,
                               label=f"anchor_check")
    user = f"Is the following visible in this photo? '{anchor}'"
    try:
        reply = gemini_caller(image_path, ANCHOR_SYS, user)
    except Exception as e:
        return False, f"ERROR: {type(e).__name__}: {e}"
    yes = bool(re.search(r"\byes\b", reply or "", re.I))
    return yes, (reply or "").strip()


# ── PASS_strict — slide 4 — a∧b∧c∧d ──────────────────────────────────
def pass_strict(format_ok, dir_ok, ckpt_ok, anchor_ok):
    """`a ∧ b ∧ c ∧ d` — slide 4. Pure."""
    return bool(format_ok and dir_ok and ckpt_ok and anchor_ok)
