"""City-specific config for the NavLM toolbox.

Each entry describes:
- search_queries: strings used by yt-dlp on fallback video sources
- landmarks: list of well-known POIs used as plausible destinations
  in auto-generated Q&A.

ANNOTATION_PROMPTS holds two prompt variants for the auto-annotator:
  - "qwen"  : free-form instruction, best for Qwen family (Qwen2.5-VL, Qwen3-VL).
  - "gemma" : markdown-structured, best for Gemma family (Gemma-4-it, Gemma-3-it).

Both variants avoid the prompt-example-leakage bug from the first draft
(the literal phrase "red awning" kept showing up in generated answers).
"""

CITIES = {
    "zurich": {
        "display_name": "Zurich, Switzerland",
        "search_queries": [
            "zurich walking tour 4k",
            "苏黎世 散步",
        ],
        "youtube_seed": "https://www.youtube.com/watch?v=_NmYvuEILw4",
        # Refined from scene-inventory pass on the Walking Tours Zurich video:
        # Qwen3-VL-32B directly identified or OCR-read most of these on frames.
        "landmarks": [
            "Zurich Hauptbahnhof (Bahnhofplatz, the main train station)",
            "Grossmünster (twin-towered stone church in the Old Town)",
            "Helmhaus Zürich (museum by the river)",
            "Lindenhof hilltop square",
            "Limmatquai (riverside promenade along the Limmat)",
            "Bellevue / Bellevueplatz area",
            "St. Peter's Church (with the tall clock tower)",
            "the Limmat river",
            "Bahnhofstrasse (main shopping street)",
            "Niederdorf pedestrian Old Town",
            "Lake Zurich",
            "Paradeplatz",
        ],
    },
    "bern": {
        "display_name": "Bern, Switzerland",
        "search_queries": ["bern walking tour 4k", "伯尔尼 散步"],
        "youtube_seed": None,
        "landmarks": [
            "Zytglogge (clock tower)",
            "Bundeshaus",
            "Bear Park",
            "Old Town (UNESCO site)",
            "Bern train station",
        ],
    },
    "amsterdam": {
        "display_name": "Amsterdam, Netherlands",
        "search_queries": ["amsterdam walking tour 4k"],
        "youtube_seed": "https://www.youtube.com/watch?v=fGX0Te6pFvk",
        "landmarks": [
            "Amsterdam Centraal",
            "Dam Square",
            "Van Gogh Museum",
            "Anne Frank House",
            "Vondelpark",
            "Rijksmuseum",
            "Museumplein",
        ],
    },
    "venice": {
        "display_name": "Venice, Italy",
        "search_queries": ["venice walking tour 4k"],
        "youtube_seed": None,
        "landmarks": [
            "St. Mark's Square",
            "Rialto Bridge",
            "Grand Canal",
            "Santa Lucia train station",
        ],
    },
    "generic": {
        "display_name": "an unknown European city",
        "search_queries": [],
        "youtube_seed": None,
        "landmarks": [
            "the nearest train station",
            "the main square",
            "the cathedral",
            "the old town",
        ],
    },
}


# ----- Qwen-style prompt v2 (anti-template) -----
# v1 tended to templatize Q2 with "I can't tell exactly from here" and always
# picked Grossmünster as the destination. v2 varies question openers,
# rotates the destination landmark, and forbids a small set of canned
# refusal phrases. This mirrors the Gemma v2 anti-template fixes.
_QWEN_PROMPT = """You are generating training data for a tourist navigation assistant. Look at the attached photograph taken by a traveler walking in {display_name}.

Generate exactly 3 conversational Q&A pairs that a real tourist might ask while holding up their phone. Each pair must follow one of these intents in this order:

1. "Where am I?" — infer the country, region, or district from visible architectural, vegetation, signage, or infrastructure cues. Quote specific visual evidence. Vary the opener across calls (e.g. "Hey, where am I?", "Any idea which country this is?", "What city is this?", "Where have I ended up?").

2. "How do I get to <LANDMARK>?" — pick ONE landmark from this list, ROTATING so you don't always pick the same one: {landmarks_list}. Vary the question phrasing each time ("How do I get to <X>?", "Which way is <X>?", "Can you point me toward <X>?", "Is <X> far from here?"). In the answer:
   (a) First try to extract a real directional cue from the scene: a street heading somewhere, a river bank, ground slope, tram direction, crowd flow, a distant skyline element, or any sign visible in the image. If you find one, base the direction on it and explain the cue ("The street slopes down to the right — if the lake is that way, so is <X>").
   (b) If there is genuinely NO usable cue, decline with a VARIED phrasing. Do NOT use "I can't tell exactly from here". Pick a different opener each time from a pool like: "Nothing in this view gives me a bearing — try turning around", "Honestly I'd need a sign or landmark", "This angle is too close-up — step back a bit", "Just a guess from this frame", "Can't orient without a sign — try the next intersection."

3. "What do I see around me?" — describe the scene in plain tourist language, referencing at least TWO distinct items actually present in the image (setting + a notable detail). Vary the opener ("What am I looking at?", "Describe what's in front of me", "What's around me?").

Hard constraints:
- Every claim must be anchored to at least one visible element.
- Do NOT invent street names, distances, shops, or routes.
- Use spoken-English phrasing; these will be spoken out loud.
- The three answers must cite DIFFERENT visual cues (no reusing the same landmark phrase across answers).
- Forbidden literal phrases: "I can't tell exactly from here", "there aren't any clear street signs or landmarks visible", any answer beginning with "I can't tell".
- Prefer a grounded best-guess over a confident fabrication; prefer a varied honest decline over a templated "I can't tell".

Output ONLY a JSON array. No markdown code fences, no prose before or after, no trailing commentary:

[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""


# ----- Gemma-style prompt v2 (anti-template) -----
# v1 produced Q2 answers that all started "I can't tell exactly from here,
# as there aren't any clear street signs...". v2 forbids that opener and
# requires the model to reason from whatever partial cues ARE visible
# (street direction, river flow, crowd, tram direction, terrain).
_GEMMA_PROMPT = """# Role
You are an annotation model generating training data for a conversational tourism navigation assistant.

# Context
A tourist is walking in {display_name} and holding up a smartphone camera. The attached image is what the camera currently sees.

# Task
Produce exactly 3 question-answer pairs. Each pair is a natural spoken interaction.

# Required intents (one per pair, in this order)

## Intent 1 — Localization
- Question: vary the opener each call (e.g. "Hey, where am I?", "Any idea which country this is?", "What city is this?", "Where exactly have I ended up?").
- Answer: identify the country / region / district, grounded in one or more visible cues (architecture, vegetation, signage language, vehicle plates, sky, terrain). Name the cue explicitly.

## Intent 2 — Direction  (the hardest one — make it non-templated)
- Destination: pick ONE landmark from this list: {landmarks_list}. Rotate among landmarks across the three calls you make for this frame's sibling calls — do not always pick the same one.
- Question opener: vary each time ("How do I get to <X>?", "Which way is <X> from here?", "I want to go to <X> — which direction?", "Is it far to <X>?", "Can you point me toward <X>?").
- Answer requirements — READ CAREFULLY:
  * First try to extract ANY partial directional cue from the visible scene: a street heading into the distance, a river (which way the water flows or which bank you're on), slope of the ground (uphill/downhill), a tram direction sign, the flow of other pedestrians, a distant skyline element (tower, mountain, lake), a compass-giving sign.
  * If you find such a cue, base your direction on it — even if uncertain — and explain the cue: "The street slopes down to the left; most Old Town routes toward X in Zurich tend to follow that direction, so try that way."
  * If you honestly find NO usable cue, decline WITH VARIETY. Do NOT start with "I can't tell exactly from here". Instead pick a different phrasing each call from this pool (or invent a similar one):
      - "Nothing in view gives me a clear bearing — mind turning around so I can see more?"
      - "Honestly I'd need a sign or a landmark; maybe ask the next local."
      - "This view is too close-up; try stepping back or walking a few meters."
      - "I'd just be guessing from this angle."
      - "Can't orient without a street sign — try the next intersection."
  * NEVER use the phrase "I can't tell exactly from here" verbatim. NEVER say "there aren't any clear street signs or landmarks visible" verbatim.

## Intent 3 — Scene description
- Question: vary the opener ("What am I looking at?", "Describe what's in front of me.", "What's around me?", "What do I see here?").
- Answer: describe the scene in tourist-friendly language, referencing at least two distinct visible elements (don't focus on a single small object). Mention the overall setting first, then a notable specific detail.

# Hard constraints
- No invented street names, distances, shops, or routes.
- Each answer must cite at least one visible element.
- The three answers must cite DIFFERENT visual cues (no reusing the same landmark/sign across answers).
- Use spoken casual English, not written formal English.
- Forbidden literal phrases:
    - "I can't tell exactly from here"
    - "there aren't any clear street signs or landmarks visible"
    - any phrase that starts with "I can't tell"

# Output format
Return a single JSON array with exactly three items. No markdown code fences. No prose before or after.

[{{"question":"...","answer":"..."}}, {{"question":"...","answer":"..."}}, {{"question":"...","answer":"..."}}]
"""


ANNOTATION_PROMPTS = {
    "qwen": _QWEN_PROMPT,
    "gemma": _GEMMA_PROMPT,
}

# Backward-compat alias — old code used ANNOTATION_SYSTEM_PROMPT
ANNOTATION_SYSTEM_PROMPT = _QWEN_PROMPT
