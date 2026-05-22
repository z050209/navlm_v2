# NavLM training prompt — draft v2

Replaces `toolbox/synth/prompts.py` SYSTEM_PROMPT and USER_TEMPLATE.

## What changed in v2

1. **User's question stays natural language** ("How do I get to Paradeplatz?",
   "I'm trying to find Großmünster", "Where is the train station?"). The
   model has to **parse it** to identify the destination, then look up
   the destination's coordinates from the nearby-POI table.
2. **CoT narrates the user's perspective** in natural order:
   "I'm at X, want to go to Y, OSM says route is Z, distance is N, I
    need to figure out which way I'm facing first, then translate."

## SYSTEM_PROMPT

```
You are a walking-direction assistant for travelers who have trouble
reading map apps. The user sends you (a) a photo from their phone
camera, (b) their current GPS, (c) a list of nearby POIs with absolute
coordinates, (d) an OSM-planned walking route in absolute compass
bearings, and (e) a natural-language question.

You DO NOT receive the camera's heading. You must figure out which way
the camera is facing by looking at the image and reasoning about which
visible landmarks sit where on the map.

Output two parts in order: <thinking> ... </thinking> then <answer>
... </answer>. The <thinking> block MUST follow the seven labelled
steps below. Two structured fields must each appear on their own line
in the exact format shown — they are parsed mechanically:

  INFERRED_HEADING: <integer 0-359>
  FIRST_ACTION: <one of: continue ahead | turn left | turn right | turn around>

Step structure inside <thinking>:

  STEP 1 (understand the question): restate in one sentence what the
    user is asking. Identify the destination they named.

  STEP 2 (resolve coordinates): state the user's current GPS, look up
    the destination in the nearby-POI list, and write its GPS. Quote
    the total walking distance and estimated minutes from the route
    block. (Format: "I'm at (lat, lon). Destination <name> is at
    (lat, lon). About N m, K minutes walking.")

  STEP 3 (read the planned route): summarise the OSM route in one or
    two sentences — the first-segment bearing and street name, plus
    a high-level shape ("two more turns then arrive" / "long route,
    six more turns").

  STEP 4 (look at the image): list visible POIs, marking each as on
    the LEFT, CENTER, or RIGHT of the frame, plus depth cues if useful
    (close vs far, blocking the view).

  STEP 5 (cross-reference image with map): for each POI listed in
    STEP 4, state which compass direction it sits relative to the
    user's GPS — "Limmat is east of me", "Großmünster is north" —
    using the coordinates from the nearby-POI list.

  STEP 6 (triangulate my heading): combine LEFT/CENTER/RIGHT (STEP 4)
    with cardinal directions (STEP 5) to deduce which way the camera
    is facing. Reason explicitly: if a POI is on the LEFT and the map
    says it is to the north, then north is on my left, so I face east.
    End STEP 6 with the line:
        INFERRED_HEADING: <integer 0-359>

  STEP 7 (translate route bearing into a relative verb):
    diff = (route_bearing - INFERRED_HEADING + 540) mod 360 - 180
    if |diff| ≤ 35° → continue ahead
    elif |diff| > 135° → turn around
    elif diff < 0 → turn left
    else → turn right
    Show this arithmetic explicitly. End STEP 7 with the line:
        FIRST_ACTION: <verb>

Then the <answer> block, 2-4 sentences total. Hard rules for the
answer:

  - NEVER use compass words (north, south, east, west, NE, SSE, ...)
  - NEVER use any numeric distance (metres, blocks, km).
  - NEVER repeat raw GPS coordinates.
  - Use the action verb chosen in STEP 7 in the FIRST sentence.
  - Reference at least one visible object from STEP 4 to ground the
    direction.
  - For routes with more than 3 turns, the answer must end by asking
    the user to send another photo when they reach a specific visible
    checkpoint, instead of describing every turn:
       "When you reach <visible checkpoint>, send me another photo and
        I'll guide you from there."
```

## USER_TEMPLATE

```
You are at GPS ({USER_LAT:.5f}, {USER_LON:.5f}).

Nearby POIs (with their absolute coordinates so you can place them on
a mental map relative to where you are):

{NEARBY_POI_LIST}

Planned walking route (absolute compass bearings — you will translate
to left/right/ahead based on which way you are facing):

  - First segment: bearing {FIRST_SEG_BEARING:.0f}° ({FIRST_SEG_COMPASS})
    for ~{FIRST_SEG_DISTANCE:.0f} m along {FIRST_SEG_STREET}.
  - Total distance: {TOTAL_DIST_M:.0f} m, ~{ESTIMATED_MINUTES} minutes.
  - Total number of turns on the route: {N_TURNS}
{REMAINING_ROUTE_BLOCK}

User asks: "{USER_QUESTION}"
```

`{NEARBY_POI_LIST}` is a 5–8 entry bulleted list of POIs within ~300 m
of the user, drawn from the OSM tier-1 set + scenery POIs. Each entry
includes the kind label so the model can use natural descriptions in
the answer. Crucially, the destination POI named in the user's question
**must be in this list** — the model will resolve it by name.

```
  - Bahnhofstrasse (the main shopping street): (47.37367, 8.53924)
  - Paradeplatz (a public square): (47.36968, 8.53887)
  - Jelmoli (a department store): (47.37480, 8.53850)
  - Fraumünster (a church with green spire): (47.37018, 8.54077)
  - Limmat river: (47.37050, 8.54300)
```

`{REMAINING_ROUTE_BLOCK}` for short routes (≤3 turns):
```
  - Then turn onto Talacker for ~120 m.
  - Then arrive at Paradeplatz.
```

For long routes (>3 turns):
```
  - The route then has 5 more turns; your answer must NOT describe
    each one. Pick a visible landmark on the FIRST segment and ask the
    user to send another photo when they get there.
```

`{USER_QUESTION}` is the natural-language ask, e.g. one of:
- "I'm trying to find Paradeplatz."
- "How do I get to Großmünster from here?"
- "Where's the main train station?"
- "I want to go to the lake."
- "Which way to Bahnhofstrasse?"

---

## Worked example (passing CoT + answer)

User message:
```
You are at GPS (47.37367, 8.53924).

Nearby POIs:
  - Bahnhofstrasse (the main shopping street): (47.37367, 8.53924)
  - Paradeplatz (a public square): (47.36968, 8.53887)
  - Jelmoli (a department store): (47.37480, 8.53850)
  - Fraumünster (a church): (47.37018, 8.54077)
  - Limmat river: (47.37050, 8.54300)

Planned walking route:
  - First segment: bearing 152° (SSE) for ~80 m along Bahnhofstrasse.
  - Total distance: 444 m, ~6 minutes.
  - Total number of turns on the route: 2
  - Then turn onto Talacker for ~120 m.
  - Then arrive at Paradeplatz.

User asks: "I'm trying to find Paradeplatz."
```

Assistant response:
```
<thinking>
STEP 1 (understand the question): The user wants directions to
Paradeplatz from where they are now.

STEP 2 (resolve coordinates): I'm at (47.37367, 8.53924). Paradeplatz
is in the nearby-POI list at (47.36968, 8.53887). The route says about
444 m, around 6 minutes walking.

STEP 3 (read the planned route): OSM's first segment goes 152° (SSE)
for about 80 m along Bahnhofstrasse, then one more turn onto Talacker
before arriving at Paradeplatz. Two turns total — short route, I can
describe both.

STEP 4 (look at the image): I see Bahnhofstrasse's tram tracks running
straight ahead, slightly to the LEFT of frame. RIGHT side is a glass
storefront (matches a Bahnhofstrasse shop). No church towers visible.
The Limmat river is not visible in the picture.

STEP 5 (cross-reference image with map): Looking at the POI
coordinates: Bahnhofstrasse runs roughly N-S through my location.
Jelmoli at (47.37480, 8.53850) is to the north of me. Paradeplatz at
(47.36968, 8.53887) is to the south-southeast. Fraumünster is south-
southeast, Limmat river is to the east.

STEP 6 (triangulate my heading): The Bahnhofstrasse tracks run roughly
straight ahead and slightly to my left. Bahnhofstrasse is a north-
south street, so "ahead-and-slightly-left" along it means I'm
facing toward the south. Limmat river is east of me but not visible
in the frame — it would be on my left if I were looking south, or
behind me if I'm looking south-southwest. The fact that it's not
visible at all (likely behind/blocked) is consistent with facing
south-southwest. So I face approximately 188°.
INFERRED_HEADING: 188

STEP 7 (translate route bearing into a relative verb):
  diff = (152 − 188 + 540) mod 360 − 180 = −36°
  |−36| > 35°, so not "continue ahead".
  |−36| ≤ 135°, so not "turn around".
  −36 < 0, so turn left.
FIRST_ACTION: turn left
</thinking>

<answer>
Turn slightly left so the tram tracks of Bahnhofstrasse line up to your
left side, and walk along them a short way. When the tracks bend, take
the side street on your left toward the open plaza. That plaza is
Paradeplatz.
</answer>
```

Why this passes our verifier:
- `INFERRED_HEADING: 188` matches ground-truth heading 188.1° → check 5 PASS.
- `FIRST_ACTION: turn left` matches planner's first_action → check 6 PASS.
- Closed-loop δ ≈ 54° (boundary case, within 55° discretization tolerance) → check 4 PASS.
- Answer has no compass words, no metres, references Bahnhofstrasse tracks → check 2 PASS.
- 4 sentences in answer → check 3 PASS.
- Both required CoT fields present and parseable → check 1 PASS.

---

## Long-route example (>3 turns, multi-turn pattern)

User asks: "How do I get to the train station?"

Route has 7 turns over 1.2 km.

Assistant response (abbreviated):
```
<thinking>
STEP 1: User wants directions to the train station (Hauptbahnhof).
STEP 2: I'm at (47.36X, 8.54X). Hauptbahnhof at (47.3779, 8.5402),
        about 1200 m, ~15 minutes walking.
STEP 3: First segment goes 25° (NNE) for ~90 m along Limmatquai.
        7 turns total — too long to describe in full; I'll describe
        only the first leg and a clear visible checkpoint.
STEP 4: I see the Limmat river right in front of me, with a stone
        bridge slightly to the LEFT. Old buildings line the
        right-hand bank.
STEP 5: Limmatquai runs along the river, which is to my north here.
        The bridge to the LEFT (Münsterbrücke at 47.370, 8.542) is
        roughly to my west.
STEP 6: West is on my LEFT → I face north.
        INFERRED_HEADING: 0
STEP 7: diff = (25 − 0 + 540) mod 360 − 180 = 25°. |25| ≤ 35° →
        continue ahead.
        FIRST_ACTION: continue ahead
</thinking>

<answer>
Walk straight ahead, keeping the river on your left. You'll soon see a
big stone bridge with arches on your left side. When you reach that
bridge, send me another photo and I'll guide you from there.
</answer>
```

Note the answer ends with the "send another photo at <visible
checkpoint>" pattern because the route is long.

---

## Step 12 verifier (unchanged from v1)

After Gemma generates each (thinking, answer) pair:

1. Parse `INFERRED_HEADING:` and `FIRST_ACTION:` from CoT.
2. Format checks on `<answer>` — no forbidden tokens.
3. Sentence count check (2-4 sentences).
4. Closed-loop angle:
   `δ = |angle_diff(inferred_heading + ACTION_DELTA[first_action],
                     first_seg_bearing_truth)|`
   PASS if δ ≤ 55°.
5. Heading sanity:
   `|angle_diff(inferred_heading, heading_gt)| ≤ 30°`.
6. Action consistency:
   `first_action == planner.first_action`.

Sample passes only if all six checks succeed.
