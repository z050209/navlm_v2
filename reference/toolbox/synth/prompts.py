"""Prompt templates for the unified synth pipeline (v3 — heading-given).

This file holds the SYSTEM_PROMPT and USER_TEMPLATE for the **v3 prompt**:
the user message DOES include the camera heading, and the model is
asked to do the (heading − bearing → relative verb) math directly.
This produced `synth_v3_train.jsonl` and trained `lora_zurich_v3` (EXP-C1).

v4a / v4b / v4c (the no-heading variants used in EXP-C2 / C3 / C4) are
DERIVED from v3 outputs by `pipeline/build_v4_datasets.py` and
`pipeline/build_v4c_rationale.py`. Their system prompts live inside
those builder scripts; this file is NOT used for v4*.

Output structure (parsed by `pipeline/step_12_closed_loop_verify.py`):
  <thinking>   six labelled steps including STEP 3 explicit math    </thinking>
  <answer>     2-4 short sentences, TTS-friendly, no compass/metres </answer>

NB: an older docstring here said "v2 — heading-from-image". That was
inaccurate; the actual prompt below is v3 (heading-given).
"""


SYSTEM_PROMPT = """You are a walking-direction assistant for travelers who have trouble reading map apps. The user sends you (a) a photo from their phone camera, (b) their current GPS, (c) the camera's compass heading (the direction the user is facing, in degrees, where 0° = north, 90° = east), (d) a list of nearby POIs with absolute coordinates, (e) an OSM-planned walking route described by the absolute bearing of its first segment, and (f) a natural-language question.

Your job is: take the user's heading and the route's first-segment bearing, compute the relative action verb (continue ahead / turn left / turn right / turn around) the user must take, then look at the photograph, pick a visible object to anchor that action to, and write a short scene-anchored answer.

Output two parts in order: <thinking>...</thinking> then <answer>...</answer>. The <thinking> block MUST follow the six labelled steps below.

Steps:

  STEP 1 (understand the question): restate in one sentence what the user is asking. Identify the destination they named.

  STEP 2 (resolve coordinates): state the user's current GPS and heading, look up the destination's GPS in the nearby-POI list, quote total distance and estimated minutes.

  STEP 3 (compute the relative action): show this arithmetic explicitly:
       diff = (route_bearing - user_heading + 540) mod 360 - 180
       if |diff| <= 35°       -> continue ahead
       elif |diff| > 135°     -> turn around
       elif diff < 0          -> turn left
       else                   -> turn right
    Write the diff value in degrees and the verb you derive from it.

  STEP 4 (look at the image): list 3-5 things you actually see in the photograph — buildings, signs, streetscape, parked vehicles, vegetation, etc. Mark each as on the LEFT, CENTER, or RIGHT of the frame.

  STEP 5 (pick an anchor): from STEP 4, choose ONE distinctive object to anchor the directions to. It should be something the user can easily identify ("the glass storefront on your right", "the tram tracks running ahead").

  STEP 6 (plan the answer): write one sentence using the action verb from STEP 3 and the anchor from STEP 5. Note whether the route is short (≤3 turns: describe the rest in the answer) or long (>3 turns: end the answer with "send me another photo at <visible checkpoint>"). For long routes, the checkpoint must be a permanent landmark from STEP 4 (a building, sign, statue, fixed feature) — NOT a movable object (vehicle, bicycle, pedestrian, café table, etc.) since the user will only arrive there minutes later.

Then the <answer> block, 2-4 short sentences. The answer will be read aloud by a text-to-speech engine, so it must sound natural when spoken:

  - NO em-dashes (—) or parentheses; use commas + "that's the X" instead:
      BAD:  "Bahnhofstrasse — the main shopping street — runs north."
      GOOD: "Bahnhofstrasse, that's the main shopping street, runs north."
  - NO bullet points, line breaks, or list formatting.
  - Use casual spoken connectors: "then", "after that", "you'll see".
  - Address the user as "you" in second person.
  - Keep individual sentences short (≤ 20 words). Long sentences are hard to follow in audio.
  - Contractions are fine and preferred ("you'll", "that's", "it's").
  - Avoid abbreviations ("st." → "street").
  - NEVER use compass words (north, south, east, west, NE, SSE, ...).
  - NEVER use any numeric distance (metres, blocks, km).
  - NEVER repeat raw GPS coordinates.
  - Use the action verb you derived in STEP 3 in the FIRST sentence.
  - Reference the anchor object you chose in STEP 5 to ground the direction.
  - The first time you name any landmark, street, square, river, or bridge, attach a brief descriptor using a comma + "that's the X" pattern (NOT em-dashes or parentheses):
      GOOD: "Bahnhofstrasse, that's the main shopping street with tram tracks"
      GOOD: "Paradeplatz, a small public square"
  - For routes with more than 3 turns, the answer must end with: "When you reach <checkpoint>, send me another photo and I'll guide you from there." Choose the checkpoint in this priority order:
      PRIORITY 1 (preferred): the named-street checkpoint provided to you in the route block under "PREFERRED CHECKPOINT". Use the street name verbatim. The user will see a street sign when they get there, so this is the most reliable checkpoint.
      PRIORITY 2 (fallback): a permanent landmark visible in the photograph (building, sign, statue, fountain). Use this only when no PREFERRED CHECKPOINT was provided.
      AVOID: movable or temporary objects — parked cars, bicycles, pedestrians, market stalls, café-table signs, advertising boards on wheels. The user only reaches the checkpoint several minutes later, so movable objects may be gone. (Movable objects are still fine as the immediate anchor in the first sentence; the constraint is only on the multi-turn checkpoint.)"""


USER_TEMPLATE = """You are at GPS ({USER_LAT:.5f}, {USER_LON:.5f}).
Your camera is facing compass heading {USER_HEADING:.0f}° (where 0=north, 90=east, 180=south, 270=west).

Nearby POIs (with absolute coordinates):

{NEARBY_POI_LIST}

OSM-planned walking route:

  - First segment absolute bearing: {FIRST_SEG_BEARING:.0f}° along {FIRST_SEG_STREET} for ~{FIRST_SEG_DISTANCE:.0f} m.
  - Total distance: {TOTAL_DIST_M:.0f} m, ~{ESTIMATED_MINUTES} minutes.
  - Total number of turns on the route: {N_TURNS}
{REMAINING_ROUTE_BLOCK}

User asks: "{USER_QUESTION}\""""


QUESTION_TEMPLATES = [
    "How do I get to {dest} from here?",
    "I want to go to {dest}. Which way?",
    "Take me to {dest}, please.",
    "Where's {dest}?",
    "Can you direct me to {dest}?",
    "I'm trying to find {dest}. How?",
]


# POI tier keywords — used by sampling.poi_tier and downstream weighting.
TIER_KEYWORDS = {
    1: [  # iconic — what tourists actually ask about
        "hauptbahnhof", "grossmünster", "grossmunster", "fraumünster", "fraumunster",
        "bellevue", "paradeplatz", "lindenhof", "opernhaus", "kunsthaus",
        "limmatquai", "bahnhofstrasse", "niederdorf", "sechseläutenplatz",
        "sechselautenplatz", "rathaus", "stadthaus", "münsterhof", "munsterhof",
        "landesmuseum", "swiss national museum", "uetliberg", "polyterrasse",
        "bürkliplatz", "burkliplatz", "zoo zurich", "fifa museum",
    ],
    2: [  # mid-tier — recognizable but secondary
        "museum", "kirche", "platz", "brücke", "brucke", "park", "garten",
        "universität", "universitat", "hochschule", "bibliothek",
    ],
}


# Verifier prompt for visual_verify_poi (in verifier.py). Lives here so all
# prompts are co-located.
VERIFIER_PROMPT_TEMPLATE = (
    "A user's GPS-derived location says they are at {poi!r} in Zurich.\n"
    "Look at this photograph and judge whether the SCENE is consistent "
    "with that claimed location. Consider buildings, signs, architecture, "
    "street layout, surrounding landmarks — NOT whether the literal word "
    "{poi!r} appears in the photo (text in the photo may be from "
    "directional/parking signs that point AWAY from where the user is).\n\n"
    "Answer with exactly one of:\n"
    "  YES - then a short phrase saying what about the scene matches "
    "{poi!r} (architecture, neighbouring landmarks, street type).\n"
    "  NO - then a short phrase describing where the scene actually "
    "looks like instead (e.g. 'a residential side street', 'this is at "
    "the train station, not at Globus').\n"
    "  UNSURE - only if the photo is too cropped/dark/blurry to judge."
)


# ────────────────────────────────────────────────────────────────────
# Helpers used by synth_unified to fill the v2 USER_TEMPLATE
# ────────────────────────────────────────────────────────────────────

def bearing_to_compass(b):
    """152° -> 'SSE'.  16-point compass."""
    points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return points[int((b % 360) / 22.5 + 0.5) % 16]


def format_nearby_poi_list(pois_dict, user_gps, max_n=8, max_radius_m=350):
    """Format up to max_n POIs within max_radius_m of user as bullet lines.

    Each entry uses the kind_label so the model can pull descriptors into
    the answer.  The destination POI must already be in pois_dict; the
    caller is expected to inject it if it falls outside the radius.
    """
    import math
    def hav(lat1, lon1, lat2, lon2):
        R = 6371000
        dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
        return 2 * R * math.asin(math.sqrt(a))

    cands = []
    for name, p in pois_dict.items():
        d = hav(user_gps[0], user_gps[1], p["lat"], p["lon"])
        if d <= max_radius_m:
            cands.append((d, name, p))
    cands.sort()
    cands = cands[:max_n]

    lines = []
    for _, name, p in cands:
        kind = p.get("kind_label") or ""
        kind_str = f" ({kind})" if kind else ""
        lines.append(f"  - {name}{kind_str}: ({p['lat']:.5f}, {p['lon']:.5f})")
    return "\n".join(lines)


def format_remaining_route(plan_human_steps, plan_struct_steps, n_turns):
    """For ≤3-turn routes, list each remaining step; for >3, give the next
    named-street transition explicitly so the model can use it as the
    multi-turn checkpoint.

    plan_human_steps : list[str] — output of planner.to_human(plan), e.g.
                       "Turn left onto Bahnhofstrasse and walk a few blocks."
    plan_struct_steps: list[dict] — plan["steps"] from way_planner;
                       each has 'action' and 'street'.
    """
    if n_turns <= 3 and len(plan_human_steps) > 1:
        rest = []
        for s in plan_human_steps[1:]:
            rest.append(f"  - Then {s}")
        return "\n".join(rest)
    elif n_turns > 3:
        # Mode B: pick the FIRST named transition after segment 0 as the
        # preferred multi-turn checkpoint. Fall back to a generic visible
        # landmark instruction if no named transition is available.
        next_named = None
        for s in plan_struct_steps[1:]:
            street = s.get("street", "")
            if street and street != "unnamed path":
                next_named = street
                break
        if next_named:
            return ("  - The route then has more turns; do NOT describe each one. "
                    "PREFERRED CHECKPOINT (use this in your answer): "
                    f"{next_named} — when the user reaches this street they "
                    "will see its street sign. Ask them to send another photo "
                    f"when they reach {next_named}. If for some reason this is "
                    "not appropriate, fall back to a permanent landmark visible "
                    "in the FIRST segment.")
        return ("  - The route then has more turns; do NOT describe each one. "
                "No clearly named next street is available. Pick a permanent "
                "landmark visible in the FIRST segment (a building, sign, or "
                "fixed structure) and ask the user to send another photo when "
                "they reach it.")
    return ""
