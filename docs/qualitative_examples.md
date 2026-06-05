# Qualitative success/failure examples
Source: `eval_pull/ablation_20260602_054707/` (2026-06-02/-03 ablation)

For each headline condition, 2 success rows (`PASS=True`) + 2 failure rows (`PASS=False`). Examples diversified across GT verbs where possible.

## `zs-heading-given`  (n=320, PASS=44.7%)

### Successes (model verb == GT verb)

#### Success #1

- **frame**: `most_famous/frame_00469.jpg`
- **destination**: Opernhaus  (GT heading=0°)
- **GT verb**: `turn right` · **model verb**: `turn right` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  You're currently facing north, and the route heads east-northeast. This means you need to turn right to follow the route.
  ```
- **<answer>**: `Look at the Opernhaus ahead. Turn right.`

#### Success #2

- **frame**: `most_famous/frame_01191.jpg`
- **destination**: Limmatquai  (GT heading=0°)
- **GT verb**: `continue ahead` · **model verb**: `continue ahead` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  You're currently facing north, and the route heads north as well. Since you're already on Niederdorfstrasse, which aligns with the direction of the route, you should continue straight ahead.
  ```
- **<answer>**: `Continue ahead.`

### Failures (model verb != GT verb)

#### Failure #1

- **frame**: `looks_perfect/frame_02080.jpg`
- **destination**: Helmhaus  (GT heading=90°)
- **GT verb**: `continue ahead` · **model verb**: `turn left` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  You're currently facing east, and the route heads northeast. Since the route is turning to the northeast, you should turn left to follow the correct direction.
  ```
- **<answer>**: `Look at the Grossmünster on your left. Turn left.`

#### Failure #2

- **frame**: `zurich_main/frame_03050.jpg`
- **destination**: Opernhaus  (GT heading=270°)
- **GT verb**: `turn around` · **model verb**: `turn left` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  You're currently facing west, and the route is heading east. This means you need to turn left to follow the correct direction towards the Opernhaus.
  ```
- **<answer>**: `Look at the buildings on your right. Turn left.`

---

## `trained-heading-given_r16_e5`  (n=320, PASS=98.4%)

### Successes (model verb == GT verb)

#### Success #1

- **frame**: `looks_perfect/frame_02080.jpg`
- **destination**: Helmhaus  (GT heading=90°)
- **GT verb**: `continue ahead` · **model verb**: `continue ahead` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  I am facing 90° (east) and the route's first segment is 78° (east-northeast). The difference is only 12°, which is almost straight ahead, so the correct action is to continue ahead.
  ```
- **<answer>**: `Can you see Münsterbrücke directly ahead?
Continue ahead.`

#### Success #2

- **frame**: `zurich_main/frame_03050.jpg`
- **destination**: Opernhaus  (GT heading=270°)
- **GT verb**: `turn around` · **model verb**: `turn around` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  I am facing 270° (west), but the route's first segment heads 83° (east). This is a difference of approximately 180 degrees, so I need to turn around.
  ```
- **<answer>**: `There is no clear landmark in front of you. Turn around.`

### Failures (model verb != GT verb)

#### Failure #1

- **frame**: `most_elegant/frame_01753.jpg`
- **destination**: Bürkliplatz  (GT heading=50°)
- **GT verb**: `turn around` · **model verb**: `turn right` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  I am facing 50° (northeast), but the route's first segment heads 186° (south). This requires a clockwise turn of 136°, which is best described as a sharp right turn.
  ```
- **<answer>**: `Look at the twin towers of the Grossmünster across the Limmat river.
Turn right.`

#### Failure #2

- **frame**: `most_elegant/frame_01736.jpg`
- **destination**: Helmhaus  (GT heading=218°)
- **GT verb**: `turn left` · **model verb**: `turn around` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  I am facing 218° (southwest), but the route's first segment heads 84° (east). This requires a counter-clockwise rotation of 134°, which is closer to a 180° turn than a 90° turn, so the correct action is to turn around.
  ```
- **<answer>**: `Can you see the green spire of Fraumünster in the distance?
Turn around.`

---

## `zs-heading-derived`  (n=265, PASS=26.8%)

### Successes (model verb == GT verb)

#### Success #1

- **frame**: `looks_perfect/frame_00187.jpg`
- **destination**: St. Peter  (GT heading=0°)
- **GT verb**: `continue ahead` · **model verb**: `continue ahead` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (visual cues): I can see Bahnhofstrasse with tram tracks running along it.
STEP 2 (apply geography): The tram tracks run roughly north-south, indicating the camera is facing north.
STEP 3 (estimated heading): I estimate I'm facing 0° (north).
STEP 4 (route comparison): The route's first segment heads 359° (north) — that's a 29° rotation north, so continue ahead.
  ```
- **<answer>**: `Continue ahead.`

#### Success #2

- **frame**: `most_famous/frame_00241.jpg`
- **destination**: Sechseläutenplatz  (GT heading=0°)
- **GT verb**: `turn right` · **model verb**: `turn right` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (visual cues): I can see boats docked on the water and a paved path along the shore.
STEP 2 (apply geography): The Limmat river flows south-to-north, and the boats are on the west bank, indicating the camera is facing north.
STEP 3 (estimated heading): I estimate I'm facing north.
STEP 4 (route comparison): The route's first segment heads east-northeast, which is a [90°] rotation [east], so
  ```
- **<answer>**: `Look at the boats on the left. Turn right.`

### Failures (model verb != GT verb)

#### Failure #1

- **frame**: `looks_perfect/frame_00477.jpg`
- **destination**: Limmatquai  (GT heading=0°)
- **GT verb**: `turn around` · **model verb**: `turn left` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (visual cues): I can see Bahnhofstrasse with tram tracks running along it.
STEP 2 (apply geography): The tram tracks run roughly south-to-north, indicating the camera is facing north.
STEP 3 (estimated heading): I estimate I'm facing 0° (north).
STEP 4 (route comparison): The route's first segment heads 157° (south-southeast) — that's a [157° - 0° = 157°] rotation [south-southeast], so turn
  ```
- **<answer>**: `Look at the tram tracks running along Bahnhofstrasse. Turn left.`

#### Failure #2

- **frame**: `most_famous/frame_00464.jpg`
- **destination**: Limmatquai  (GT heading=0°)
- **GT verb**: `continue ahead` · **model verb**: `turn left` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1: I can see the Opernhaus (Opera House) and Sechseläutenplatz.
STEP 2: The Opera House is on the right side, indicating the Limmat is on my right, so I'm facing roughly south.
STEP 3: I estimate I'm facing 180° (south).
STEP 4: The route's first segment heads 345° (north-northwest). That's a 145° rotation west, so turn left.
  ```
- **<answer>**: `Look at the Opera House on your right. Turn left.`

---

## `trained-heading-derived_r16_e5`  (n=265, PASS=68.7%)

### Successes (model verb == GT verb)

#### Success #1

- **frame**: `looks_perfect/frame_00187.jpg`
- **destination**: St. Peter  (GT heading=0°)
- **GT verb**: `continue ahead` · **model verb**: `continue ahead` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (visual cues): I can see a wide, straight street with tram tracks running down the middle. On the left is a large, modern glass building with "KUONI" written on it. This is clearly Bahnhofstrasse.
STEP 2 (apply geography): Bahnhofstrasse runs on a north-south axis. The Hauptbahnhof (main station) is at the north end, and Lake Zurich is at the south end. The long, straight view suggests I am
  ```
- **<answer>**: `Notice Bahnhofstrasse with the tram tracks stretching out before you.
Continue ahead.`

#### Success #2

- **frame**: `most_famous/frame_00241.jpg`
- **destination**: Sechseläutenplatz  (GT heading=0°)
- **GT verb**: `turn right` · **model verb**: `turn right` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (visual cues): I can see a large body of water with many sailboats, which is Lake Zurich. There is a paved promenade running alongside the water on the left. The path continues straight into the distance along the lakeshore.
STEP 2 (apply geography): Lake Zurich is at the southern end of the city. The Limmat river flows north out of it. Since I am looking along the shore towards the open wa
  ```
- **<answer>**: `Look at Lake Zurich stretching ahead of you.
Turn right.`

### Failures (model verb != GT verb)

#### Failure #1

- **frame**: `saturday_morning/frame_00533.jpg`
- **destination**: Münsterhof  (GT heading=180°)
- **GT verb**: `turn right` · **model verb**: `turn left` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (visual cues): I can see the Limmat river on my left, the cobblestone path of Limmatquai I am standing on, and the Münsterbrücke bridge ahead of me. The sun is high and bright, casting strong shadows.
STEP 2 (apply geography): The Limmat river flows south-to-north. Since the river is on my left and I am looking towards the bridge, I must be on the east bank of the river, looking north. The 
  ```
- **<answer>**: `Look at the Münsterbrücke bridge crossing the Limmat river just ahead of you.
Turn left.`

#### Failure #2

- **frame**: `zurich_main/frame_01993.jpg`
- **destination**: Lake Zurich  (GT heading=0°)
- **GT verb**: `turn left` · **model verb**: `turn right` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (visual cues): I can see the large, open cobblestone square of Sechseläutenplatz. Across the square is the grand facade of the Opernhaus Zürich. The sun appears to be high and slightly behind me, casting short shadows forward and to the left.
STEP 2 (apply geography): The Opernhaus is located at the southern end of Sechseläutenplatz. Since I am looking at it across the square, I must be sta
  ```
- **<answer>**: `Notice the Opernhaus across Sechseläutenplatz.
Turn right.`

---

## `zs-heading-implicit`  (n=267, PASS=28.1%)

### Successes (model verb == GT verb)

#### Success #1

- **frame**: `bahnhofstrasse/frame_00394.jpg`
- **destination**: Lindenhof  (GT heading=0°)
- **GT verb**: `turn right` · **model verb**: `turn right` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (what I see): The visible scene contains Bahnhofstrasse with tram tracks running along it.
STEP 2 (where the destination is relative to me): The destination is "to my right" as the tram tracks lead towards Lindenhof.
STEP 3 (verb decision): Therefore the walker should turn right.
  ```
- **<answer>**: `Look at the tram tracks curving to your right. Turn right.`

#### Success #2

- **frame**: `looks_perfect/frame_00169.jpg`
- **destination**: Rathaus  (GT heading=0°)
- **GT verb**: `continue ahead` · **model verb**: `continue ahead` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (what I see): The visible scene contains Bahnhofstrasse with tram tracks running along it.
STEP 2 (where the destination is relative to me): The Rathaus is ahead of me, as the tram tracks lead towards it.
STEP 3 (verb decision): Therefore the walker should continue ahead.
  ```
- **<answer>**: `Continue ahead.`

### Failures (model verb != GT verb)

#### Failure #1

- **frame**: `looks_perfect/frame_02182.jpg`
- **destination**: Bürkliplatz  (GT heading=180°)
- **GT verb**: `turn right` · **model verb**: `continue ahead` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (what I see): The visible scene contains Grossmünster church on the left and Limmatquai on the right.
STEP 2 (where the destination is relative to me): Bürkliplatz is ahead of me, as the tram tracks lead towards it.
STEP 3 (verb decision): Therefore the walker should continue ahead.
  ```
- **<answer>**: `Continue ahead.`

#### Failure #2

- **frame**: `zurich_main/frame_00923.jpg`
- **destination**: Münsterbrücke  (GT heading=270°)
- **GT verb**: `turn left` · **model verb**: `continue ahead` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (what I see): The visible scene shows a narrow cobblestone street with buildings on both sides.
STEP 2 (where the destination is relative to me): Münsterbrücke is ahead of me, as the street appears to lead towards it.
STEP 3 (verb decision): Therefore the walker should continue ahead.
  ```
- **<answer>**: `Continue ahead.`

---

## `trained-heading-implicit_r16_e5`  (n=267, PASS=58.4%)

### Successes (model verb == GT verb)

#### Success #1

- **frame**: `most_elegant/frame_02276.jpg`
- **destination**: Landesmuseum  (GT heading=270°)
- **GT verb**: `turn around` · **model verb**: `turn around` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (what I see): The visible scene shows the Münsterbrücke bridge leading towards the Fraumünster church with its green spire, and the St. Peter church tower is visible to the left of it.
STEP 2 (where the destination is relative to me): The Landesmuseum is located near the lake, which is to the east of this position. The view ahead is west, across the river. Therefore, the destination is behi
  ```
- **<answer>**: `Can you see Münsterbrücke directly ahead?
Turn around.`

#### Success #2

- **frame**: `looks_perfect/frame_00169.jpg`
- **destination**: Rathaus  (GT heading=0°)
- **GT verb**: `continue ahead` · **model verb**: `continue ahead` · **PASS**: ✓
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (what I see): The visible scene shows Bahnhofstrasse, a wide street with tram tracks running down the middle, lined with shops and trees.
STEP 2 (where the destination is relative to me): The initial walking route follows this street straight ahead. The destination, Rathaus, is further along this path.
STEP 3 (verb decision): Therefore the walker should continue ahead.
  ```
- **<answer>**: `Notice Bahnhofstrasse with its shops and tram tracks stretching out in front of you. Continue ahead.`

### Failures (model verb != GT verb)

#### Failure #1

- **frame**: `looks_perfect/frame_02182.jpg`
- **destination**: Bürkliplatz  (GT heading=180°)
- **GT verb**: `turn right` · **model verb**: `turn left` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (what I see): The visible scene shows the twin towers of Grossmünster on the left and the street Limmatquai stretching out ahead.
STEP 2 (where the destination is relative to me): I am facing east along the river. The destination, Bürkliplatz, is located further down the river but to the west, past the Grossmünster. Therefore, the destination is to my left.
STEP 3 (verb decision): Therefore
  ```
- **<answer>**: `Look at the Grossmünster cathedral towers on your left.
Turn left.`

#### Failure #2

- **frame**: `zurich_main/frame_00923.jpg`
- **destination**: Münsterbrücke  (GT heading=270°)
- **GT verb**: `turn left` · **model verb**: `turn right` · **PASS**: ✗
- **<thinking>** (truncated to 400 chars):
  ```
  STEP 1 (what I see): The visible scene contains a cobblestone street leading downhill between buildings, with shops on both sides. There are no major landmarks visible.
STEP 2 (where the destination is relative to me): The walking route begins by heading south, which is to my right from the current eastward-facing view.
STEP 3 (verb decision): Therefore the walker should turn right.
  ```
- **<answer>**: `There is no clear landmark in front of you. Turn right.`

---

