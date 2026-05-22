# NavLM v2 — Developer Manual

NavLM fine-tunes a small vision-language model to give **spoken walking
directions from a phone photo + GPS, without a compass** — the model
infers camera heading by triangulating visible landmarks against a
nearby-POI map. Training data is self-supervised from 8 unlabelled
YouTube Zurich walking-tour videos; the contribution is the data
pipeline, not the model.

This manual describes the method directly. `reference/` holds the old
`navlm_ss` code, cited only as an implementation reference.

---

## 1. Pipeline overview

```
videos ─▶ frames ─▶ quality filter ─▶ POI scan (Gemma — kept as-is)
                                          │
   Street View reference grid ◀───────────┘
        │
        ▼
   GPS recovery:  DINOv2 match  +  VLM place-naming  ─▶  weighted score
        │
        ▼
   OSM + HMM road-snapping ─▶ trusted (GPS, heading) frames   ◀── Phase A
        │
        ▼  Phase B
   route planning ─▶ instruction annotation (Gemini 2.5 Pro) ─▶ verify
        │
        ▼  Phase C / D
   LoRA SFT (Modal GPU)  ·  zero-shot baselines (local)  ·  eval
```

Phase A (video → trusted GPS+heading frames) is built first.

---

## ▶ How to run — commands, in order

Once per terminal, `cd` into the repo and point `python` at the shared
venv. **Do not** use the venv's `Activate.ps1` — it lives on Google
Drive, so PowerShell's execution policy blocks it as untrusted. Calling
the venv `python.exe` directly has no such issue:

```powershell
cd C:\Users\z0502\Desktop\cs231n\navlm_v2
# alias `python` to the project venv for this terminal — no Activate.ps1
function python { & "G:\My Drive\cs231n\project\cs231n\cs231n\navlm_ss\.venv\Scripts\python.exe" @args }
```

Then run these one by one. Only ✅ steps exist yet; ⏳ land as built (§7).

| # | Step | Command | Status |
|---|------|---------|--------|
| 0 | sanity-check config + paths | `python config.py` | ✅ |
| 1 | run the unit tests | `python -m pytest tests/ -q` | ✅ |
| 2 | print the 27 candidate POIs | `python -m src.poi --list` | ✅ |
| 2b| write the POI icon map | `python -m src.poi --map` | ✅ |
| 3 | list source videos found | `python -m src.extract_frames --list` | ✅ |
| 4 | extract frames — all videos | `python -m src.extract_frames` | ✅ |
| 4b| extract one video only | `python -m src.extract_frames --only hidden_streets` | ✅ |
| – | download a missing video | `python -m src.download_videos --only <name>` | ✅ (videos present) |
| 5 | Street View — derived crawl bbox | `python -m src.streetview --bbox` | ✅ |
| 5b| Street View — free metadata scan | `python -m src.streetview --scan` | ✅ |
| 5c| Street View — Static API download | `python -m src.streetview --download` | ✅ (costs $) |
| 6 | GPS recovery (DINOv2 + VLM) | `python -m src.gps_recovery` | ⏳ module coming |
| 7 | OSM + HMM road-snapping | `python -m src.road_snap` | ⏳ module coming |
| – | LoRA training on Modal | `modal run train_modal.py` | ✅ built |

Notes:
- Step 3 writes frames to `data/cities/zurich/frames/<name>/` plus an
  `extract_report.json` with per-video keep/drop counts.
- `config.py` locates `ffmpeg` automatically (installed via winget).
- Each `src` module also accepts `-h` for its options.

---

## 2. Pipeline stages

### 2.1 Video acquisition

8 YouTube walking-tour videos (`milestone2/videos/video_urls.md`);
`saturday_morning` is the evaluation hold-out. Videos already downloaded
to `videos/` — no fetch needed. `src/download_videos.py` (yt-dlp) exists
for any not yet present.

### 2.2 Frame extraction — `src/extract_frames.py`

Three steps per video; thresholds in `config.py`.

**Step 1 — dense sampling.** `ffmpeg -vf fps=1 -q:v 3` samples one JPEG
per second of video into a `<name>_dense/` cache (kept, so the filters
can be re-tuned without re-decoding).

**Step 2 — quality gate** (new — removes unusable frames):
- **Blur — variance of the Laplacian.** Convert the frame to grayscale,
  apply the Laplacian operator (a 2nd-derivative edge detector), and
  take the *variance* of the result. A sharp frame has strong, varied
  edges → high variance; a motion-blurred frame is smooth → low
  variance. Frames below `BLUR_MIN_VAR` (100) are dropped.
- **Exposure — mean luminance.** Mean grayscale value (0–255). Below
  `EXPOSURE_DARK` (25) the frame is too dark (tunnel, deep shade);
  above `EXPOSURE_BRIGHT` (230) it is blown out. Both are dropped.

**Step 3 — perceptual-hash dedup.** Each surviving frame gets a 64-bit
pHash (downscale → DCT → low-frequency coefficients thresholded at their
median) — a fingerprint robust to small changes. Greedy keep: the first
frame is always kept; a later frame is kept only if the Hamming distance
of its pHash from the *last kept* frame is ≥ `PHASH_THRESHOLD` (10 bits).
A long static stretch therefore collapses to a single representative.

Order matters: quality gate **before** dedup, so a blurry frame is never
chosen as a scene's representative.

**Code:** `src/extract_frames.py` · **In:** `videos/**/*.mp4` ·
**Out:** `data/cities/zurich/frames/<name>/frame_NNNNN.jpg` +
`frames/extract_report.json` (per-video keep/drop counts) ·
**Run:** `python -m src.extract_frames` (`--list` preview, `--only <name>`).

### 2.3 POI table

A POI is a named place with coordinates. v2 uses **one POI table**:

- **OSM-extracted points** — `extract_osm_pois.py` queries OpenStreetMap
  via osmnx Overpass over the project bbox and keeps 7 tag groups
  (tourism, historic, amenity, transport stations, leisure, place),
  filtered by name. → 453 point POIs.
  Command: `python extract_osm_pois.py --bbox 8.520,47.360,8.570,47.395`
  (W,S,E,N). The bbox is central Zurich's old town — the area the
  walking tours cover (Hauptbahnhof → Altstadt → Grossmünster → lakefront),
  ≈ 3.8 km × 3.9 km. **This bbox is the GPS scope of the project.**
- **Way / area features** — streets, the Limmat, Lake Zurich, bridges.
  These matter for navigation ("walk along Bahnhofstrasse") but OSM tags
  them as *ways / polygons*, which the point extraction above skips.
  **(Q1)** v2 does **not** hand-curate these. It pulls them from OSM
  *programmatically* — `highway` (named streets), `waterway` +
  `natural=water` (river, lake), `man_made=bridge` — keeping their real
  geometry, so "near a street/river" is a true point-to-line distance,
  not a hand-set radius.

> **Dropped:** both hand-curated tables — `zurich_landmarks_gps.py` (31
> entries, only ever used for OCR alias matching, and OCR is removed) and
> `scenery_pois.py` (13 entries with hand-set radii). v2 has **one POI
> table, fully OSM-extracted** — points + way/area features above.

**Purpose of the way / area features.** Walking instructions constantly
reference them — "walk *along Bahnhofstrasse*", "*cross Münsterbrücke*",
"follow *the Limmat*". A point landmark (a single dot) cannot represent
a street or a river, so these are kept with line/polygon geometry;
"near a street" is then a true **point-to-line distance**, not a
hand-set radius. They serve as **destinations**, as in-route
**anchors/checkpoints**, and their `kind_label` ("the main shopping
street") gives the spoken answer its natural wording.

**Code to derive the table — `src/pois.py`** (osmnx Overpass over the
project bbox):

```python
import osmnx as ox
# point landmarks
pts  = ox.features_from_bbox(bbox, tags={
    "tourism": True, "historic": True, "railway": "station",
    "amenity": ["theatre", "museum", "place_of_worship", "townhall"]})
# way / area features — kept WITH geometry
ways = ox.features_from_bbox(bbox, tags={
    "highway": ["primary", "secondary", "residential",
                "pedestrian", "living_street"],   # named streets
    "waterway": "river", "natural": "water",       # the Limmat, the lake
    "man_made": "bridge"})                         # bridges
```

Each row → `{name, aliases, kind_group, lat, lon, geometry}`; point POIs
keep a lat/lon, ways/areas keep the polyline/polygon. Output:
`data/cities/zurich/pois.json`. Run: `python -m src.pois`.

**Aliases & name resolution.** One place has many names — "ETH" /
"ETH Zürich" / "Eidgenössische Technische Hochschule". `src/pois.py`
collects the OSM alternative-name tags (`alt_name`, `short_name`,
`official_name`, `loc_name`, `name:en`, `name:de` — English/German only,
**no Chinese**) into a per-POI `aliases` list. `resolve_poi(name)`
matches a query against the name **and** its aliases — exact first, then
substring — so "how do I get to ETH?" resolves even when the table's
canonical name differs. A modern VLM already *knows* "ETH" is the
university; the alias list is what lets the deterministic POI lookup
agree. A name that still misses can be canonicalised by the geo-check VLM.

**How the POI table is used** downstream:
1. **Destination sampling** (§2.7) — annotation draws each frame's 3
   destinations from it, distance-banded.
2. **Nearby-POI list** — the VLM prompt lists POIs within range with
   coordinates, so the model can triangulate its heading against them.
3. **Place-name resolution** (§2.5) — the geo-check VLM names a place;
   it is resolved to GPS through this table.
4. **Route anchors / checkpoints** — streets and bridges named in
   multi-turn directions ("when you reach Bahnhofstrasse…").
5. **Routing targets** (§2.6) — the route planner routes to a POI.
6. **POI-region split** (§3) — the POI set is partitioned for the
   POI-generalization ablation.

**POI scan of the video frames** — `scan_video_pois_multi.py` had a VLM
(Gemma) look at frames and list visible POIs from a fixed **27-candidate
list** (`CANDIDATE_POIS`). The existing scan output
(`_video_poi_multi.jsonl`) is kept as-is — **not rerun**.

**The 27 candidates (Q2).** There is **no extraction code** — the list
was *hand-picked* (the tier-1 / "L1" iconic set, short so the VLM picks
from a manageable menu). It now lives canonically — with GPS, kind and
中文 names — in `src/poi.py`.

**Code:** `src/poi.py` · **Out:** `viz/poi_candidates_map.html` (a
signature emoji icon per POI kind) · **Run:** `python -m src.poi --list`
(table) · `python -m src.poi --map` (icon map).

| # | English | 中文 | kind | # | English | 中文 | kind |
|--|---------|------|------|--|---------|------|------|
| 1 | Hauptbahnhof | 苏黎世中央车站 | station | 15 | Stadthaus | 市政府大楼 | civic |
| 2 | Lindenhof | 林登霍夫山丘 | hill | 16 | Opernhaus | 苏黎世歌剧院 | culture |
| 3 | Paradeplatz | 阅兵广场 | square | 17 | Kunsthaus | 苏黎世美术馆 | museum |
| 4 | Münsterhof | 明斯特霍夫广场 | square | 18 | Landesmuseum | 瑞士国家博物馆 | museum |
| 5 | Fraumünster | 圣母大教堂 | church | 19 | Polyterrasse | 联邦理工观景台 | hill |
| 6 | Grossmünster | 大教堂 | church | 20 | Globus | 高乐斯百货 | store |
| 7 | St. Peter | 圣彼得教堂 | church | 21 | Jelmoli | 耶尔莫利百货 | store |
| 8 | Bellevueplatz | 贝尔维尤广场 | square | 22 | Bahnhofstrasse | 班霍夫大街 | street |
| 9 | Sechseläutenplatz | 六鸣节广场 | square | 23 | Niederdorfstrasse | 下村街 | street |
| 10 | Bürkliplatz | 比尔克利广场 | square | 24 | Limmatquai | 利马特河滨道 | street |
| 11 | Quaibrücke | 码头桥 | bridge | 25 | Rennweg | 伦韦格街 | street |
| 12 | Münsterbrücke | 大教堂桥 | bridge | 26 | Limmat river | 利马特河 | water |
| 13 | Rathausbrücke | 市政厅桥 | bridge | 27 | Lake Zurich | 苏黎世湖 | water |
| 14 | Rathaus | 市政厅 | civic | | | | |

### 2.4 Street View reference grid

The reference index against which video frames are matched is Google
Street View panoramas (`reference/fetch_streetview_grid.py`):

- The **free metadata endpoint** scans a grid and returns every panorama
  ID + exact GPS at $0.
- The **Street View Static API** ($7/1000) then downloads 4 headings
  (N/E/S/W) per panorama.

**Crawl bbox (Q3).** Yes — the grid bbox is the **bounding box of the
candidate POIs the videos visit** (the §2.3 POIs whose names appear in
`_video_poi_multi.jsonl`), **+ a ~300 m margin** so a POI on the edge,
or a route segment leaving the box, still has reference imagery. It is
*derived*, not hand-set:

```
bbox = (min_lon − m, min_lat − m, max_lon + m, max_lat + m)
       over the visited POIs,  m ≈ 300 m   →   config.SV_BBOX
```

`src/streetview.py` computes this from `src/poi.py` coordinates (later
also from recovered route GPS). The metadata scan is free, so the box
can be widened and re-crawled incrementally if routes reach an edge.

**Code:** `src/streetview.py` · **In:** `config.SV_BBOX` + Google Maps
key · **Out:** `data/cities/streetview/zurich/{images/,meta.jsonl}` ·
**Run:** `python -m src.streetview --scan` (free metadata scan) then
`--download` (Static API). ⏳ module to be built.

### 2.5 GPS recovery

Each video frame needs a recovered **GPS** *and* **heading**. Neither
DINOv2 nor a VLM is reliable enough alone, so v2 combines two
independent estimates with a weighted score.

**Estimate 1 — DINOv2 visual match.** `dinov2-base` (avg-pooled)
embeds the frame; cosine-match against the Street View index. Output:
the matched pano GPS `g_dino` and the best cosine similarity `s ∈ [0,1]`.
An absolute floor `min_sim` rejects weak matches (cosine similarity is
relative — without a floor an argmax always returns *something*).

**Estimate 2 — VLM place-naming.** The geo-check VLM (**Gemini 2.5
Pro**, Q6) is shown the frame and is *not* asked to localize precisely —
it **names a place**. It returns strict JSON:

```json
{"place": "<named street/square/landmark>",
 "lat": <decimal>, "lon": <decimal>,
 "confidence": "high|medium|low",
 "reasoning": "<one sentence>"}
```

The `place` string is resolved to GPS `g_vlm` via the POI table
(preferred over the VLM's own lat/lon, which is coarser).

**Reconciliation — weighted score (Q4).** For each frame:

- `d  = haversine(g_dino, g_vlm)` — disagreement, in metres
- `a  = exp(-d / D0)` — agreement term, `D0 ≈ 150 m` (a = 1 when the two
  estimates coincide, decaying as they diverge)
- `c` — VLM confidence as a number: low = 0.3, medium = 0.6, high = 1.0
- **combined quality** `Q = w_s·s + w_a·a + w_c·c`, weights summing to 1;
  start at `(w_s, w_a, w_c) = (0.4, 0.4, 0.2)`

A frame's GPS is **accepted if `Q ≥ τ`**. The accepted coordinate is the
confidence-weighted mean of `g_dino` and `g_vlm` when they agree,
otherwise `g_dino` when only the DINOv2 match is strong. `τ`, `D0` and
the weights are tuned by running the cheap **sample-test batch** at a
few settings and **eyeballing 10 % of the matches**; the setting that
keeps the most frames at acceptable quality wins. (Weights are heuristic
for now — with a labelled subset they could be fit by logistic
regression.)

**Heading.** Each Street View crop was rendered at a *known* heading
(0/90/180/270°). The frame's camera heading ≈ the heading of the matched
crop. **Top-k (Q4):** the DINOv2 match returns the `k` Street View crops
with the **highest cosine similarity** to the frame's embedding (`k` is
a config parameter, default `k = 5`); the heading is the
outlier-filtered **circular mean** of those k crops' rendered headings,
and the circular spread gives a confidence (geometry in
`reference/toolbox/compute_frame_heading.py`).

**Road-snapping.** The accepted per-frame GPS sequence is smoothed onto
the OSM walking graph with HMM map-matching (Newson-Krumm Viterbi),
removing jitter and forcing positions onto walkable geometry.

Phase A output: **trusted frames, each with (GPS, heading)**.

### 2.6 Routing — and turning a route into an instruction (Q5)

`way_planner.py` (osmnx + networkx) computes the OSM walking route from
a frame's GPS to a destination POI: `nx.shortest_path` by length, giving
the route geometry and the **first-segment absolute bearing `B`**.

HMM road-snapping (§2.5) gives **GPS positions only** — not which way
the camera faces. That is why the per-frame **heading** matters: the
instruction the model must produce is *relative to the camera*.

```
relative action = action_for( B − camera_heading )
   |Δ| ≤ 35°   -> "continue ahead"
   |Δ| > 135°  -> "turn around"
   Δ < 0       -> "turn left"      Δ > 0 -> "turn right"
```

So a frame whose camera faces forward while the route goes *behind* the
walker yields `|Δ| > 135°` → **"turn around"**. The picture's facing and
the route are reconciled exactly here: an absolute route bearing minus
the recovered camera heading. A frame without a trustworthy heading
cannot be turned into a relative instruction and is not used.

### 2.7 Instruction-tuning annotation

The teacher VLM (**Gemini 2.5 Pro**) is given a frame + its GPS +
heading + nearby POIs + the planned route, and produces `<thinking>`
(6 labelled reasoning steps) + `<answer>` (2–4 TTS-friendly sentences,
relative verbs anchored to a visible object).

**How each frame's samples are formed (Q6).** Per frame we draw **3
destination POIs**, sampled by a **distance-band distribution** so most
prompts are realistic short-range walks:

| Walking distance to destination | Share of the 3 |
|---------------------------------|----------------|
| ≤ 500 m (a few minutes)         | **80 %** |
| 500–1000 m                      | **10 %** |
| 1000–1500 m                     | **10 %** |

Within each band the destination is drawn tier-weighted toward iconic
POIs. So a typical frame yields ≈ 2–3 short-range instructions plus the
occasional longer one; across the dataset the 80/10/10 split holds.

Every sample is gated by a **closed-loop verifier**: parse the action
verb from the answer, check `|heading + ACTION_DELTA[verb] −
route_bearing| < 30°`. Samples that fail are dropped.

**Why 30° (Q5).** The 4 action verbs discretize heading into **4 bins of
90°** (`ACTION_DELTA = {ahead 0, left −90, right +90, around 180}`); the
boundary between "continue ahead" and a turn sits at ±45° — half a bin.
A sample is "correct" if the required turn lands inside its verb's bin,
i.e. within 45°. **30° is that 45° half-bin minus a ~15° safety margin**
— it accepts a sample only when it sits comfortably inside the right
bin, excluding borderline cases. It is a discretization-driven tolerance,
not a deep result: loosen toward 45° for more data, tighten for cleaner
labels.

**Run 5 samples first.** The annotation module takes a `--limit 5` flag;
we inspect the 5 (thinking + answer + verifier verdict) before the full
run. Code is `synth_unified.py`'s logic with the teacher swapped to
`call_gemini(model="gemini-2.5-pro")`.

### 2.8 Image ↔ POI indexing & route map

Indexes (kept): `_video_poi_multi.jsonl` (`{video, frame_id,
visible_pois[]}`) and the Street-View-side visibility index, bridged by
GPS — so any frame is traceable POI→frames and frame→POIs, which also
yields the per-POI dataset distribution.

Once frames carry GPS, plot the **8-video route map**: each video's
recovered GPS sequence as a coloured polyline on one Leaflet/folium map —
both a deliverable and a sanity check on GPS recovery.

### 2.9 Gemini API budget (Q6)

Per 1M tokens: Gemini 2.5 **Pro** $1.25 in / $10 out. Both VLM stages
run on Pro (Q6); the POI scan is *not* rerun.

| Workload | Calls | Model | Est. cost |
|----------|------:|-------|----------:|
| VLM geo-check — candidate frames | ~5,000 | Pro | ~$18 |
| Instruction annotation — 3 dest/frame | ~6,600 | Pro | ~$76 |
| Street View Static crawl | — | — | ~$54 |

Total ≈ **$148**, against a **$50 GCP credit** — over budget. Mitigate
by trimming the Street View crawl to the video routes and capping
annotation samples. Flagged for the budget owner; tracked live with
`reference/track_spend.py`.

---

## 3. Train / test — two separate ablations (Q7)

The hold-outs are **not** combined into one split — they are two
independent ablation experiments:

1. **Video / camera generalization** — train on 7 videos, test on the
   held-out video (`saturday_morning`). Tests new footage.
2. **POI generalization** — train on one set of destination POIs, test
   on a disjoint set. Tests routing to places unseen in training.

**Assigning each frame to a POI region (Q7).** For the POI split we
partition the map into **one region per POI** and label each frame by
the region its recovered GPS falls in:

- regions = a **Voronoi partition** of the POI coordinates, clipped to
  the project bbox (each frame → its nearest POI). For scenery entries
  (streets, river, lake) the region is instead that feature's `radius_m`
  disc / buffer polygon.
- a frame "belongs to" POI X if its GPS lies in X's region.
- split the **POI set** into train / test; each frame then goes to train
  or test by its region's POI — so a test-POI's frames are never trained on.

The region polygons are exported and drawn on a map (§5) so the split is
visually auditable — you can see which area, and which POI, each frame
fell into.

---

## 4. Experiments, training & evaluation

### 4.1 The question

Can a VLM give *correct* walking directions from a phone photo + GPS
**without a compass** — by inferring its camera heading from the photo?

### 4.2 Conditions

**Three prompt variants × {base, LoRA} = 6 conditions**, run for **each**
ablation split (§3):

| ID | model | heading in prompt | chain-of-thought |
|----|-------|-------------------|------------------|
| **B-given**    | base Qwen2.5-VL-7B, zero-shot | given  | — |
| **B-implicit** | base, zero-shot               | hidden | implicit — no heading step |
| **B-explicit** | base, zero-shot               | hidden | explicit `INFERRED_HEADING:` step |
| **L-given**    | + NavLM LoRA                  | given  | — |
| **L-implicit** | + NavLM LoRA                  | hidden | implicit |
| **L-explicit** | + NavLM LoRA                  | hidden | explicit `INFERRED_HEADING:` step |

The two compass-free variants differ in **how the CoT handles heading**:
- **implicit** — heading is removed from the prompt; the model just
  writes the answer and never states a heading.
- **explicit** — the `<thinking>` block has a dedicated step where the
  model writes its **`INFERRED_HEADING:`** — it triangulates visible
  landmarks against the nearby-POI map and commits to a heading number
  *before* choosing the action. This makes heading inference an
  explicit, trainable, interpretable intermediate output.

Headline comparisons:
- **L-explicit vs B-explicit** — does fine-tuning teach the model to
  triangulate its heading from the photo?
- **L-explicit vs L-implicit** — does forcing heading into the CoT help?
- **L-explicit vs L-given** — the accuracy cost of dropping the compass.

`*-given` are upper-bound references. **L-explicit** — compass-free with
explicit heading reasoning — is the main result. (In the prior round it
was the best compass-free model: median heading error 21° vs 66° for
implicit.)

### 4.3 Training

| | |
|--|--|
| Base model | Qwen2.5-VL-7B-Instruct |
| Method | LoRA SFT — r=16, α=32, dropout 0.05, target `q/k/v/o_proj` |
| Quantization | 4-bit NF4 base, BF16 adapters (~0.5 % params trainable) |
| Data | synth set (§2.7) — frame + system prompt + user msg + assistant `<thinking>`+`<answer>` |
| Schedule | 2 epochs · lr 2e-4 · cosine · 3 % warmup |
| Batch | 1 × grad-accum 8 (effective 8); images capped 448² px |
| Compute | Modal A100-80GB via `train_modal.py` — ~3–6 h, ~$22 / run |

`L-given`, `L-implicit` and `L-explicit` are three LoRA runs on the same
frames. The teacher annotates once (heading visible, §2.7); the implicit
and explicit training sets are *derived* from that base set — implicit
strips the heading from the user message, explicit additionally rewrites
the `<thinking>` to spell out the `INFERRED_HEADING:` step.

### 4.4 Evaluation

**Test set** = the held-out split of whichever ablation is running — the
`saturday_morning` video, or the held-out destination POIs (§3).

Every test frame carries **ground truth** from Phase A: its verified GPS
and heading, plus the OSM-planned route to each destination (the route's
first-segment bearing).

**Procedure.** Feed photo + GPS (+ heading for `*-given`) + nearby POIs
+ route; the model emits `<thinking>` + `<answer>`; the answer is scored
on **four named metrics**. Each (except the form check) compares the
model's output against ground truth.

**(a) Format compliance** *(model output only, no GT)*. Both
`<thinking>` and `<answer>` blocks present, the answer is 2–4 sentences,
and it contains no compass words, numbers, or raw GPS. Standard
instruction-following / output-validity check.

**(b) Directional accuracy** *(model's verb vs GT geometry)* — the core
task-correctness metric, a "closed-loop" angular check:
- *GT:* the frame's heading `h` and the route's first-segment bearing
  `B`, both known from Phase A and the route planner.
- *Test output:* parse the action verb the model wrote ("turn left",
  "continue ahead", …) out of `<answer>`.
- *Close the loop:* if the user faces `h` and performs that verb, their
  new facing is `h + ACTION_DELTA[verb]`, where
  `ACTION_DELTA = {ahead 0, left −90, right +90, around 180}`.
- For the instruction to be **correct**, that new facing must match the
  direction they actually need to go (`B`):
  `δ = | angle_diff( h + ACTION_DELTA[verb], B ) |`.
- **Correct if δ < 30°.** Saying "turn left" when the route is to the
  right gives δ ≈ 180° → wrong.

**(c) Checkpoint validity** *(model's checkpoint vs GT route)* — a route
grounding check. A long route (>3 turns) ends with "when you reach <X>,
send me another photo". `<X>` must be (a) a real street/landmark **on
the planned route** — a membership test against the route's
streets/POIs — and (b) a *permanent* feature, not a movable object
("the red car" fails). Single-turn answers skip it.

**(d) Anchor faithfulness — the hallucination metric** *(model's anchor
vs the photo)*. The answer anchors the action to a visible object —
"turn left **at the tram tracks**". This extracts the anchor phrase and
asks a VLM (Gemini) "is <anchor> visible in this image? yes/no". GT is
the photo itself. This is exactly an **object-hallucination check** — it
catches the model inventing a landmark that isn't in the frame
(the well-known VLM visual-hallucination failure mode).

**Scoring — PASS_strict, not a weighted sum.** The headline metric is
**`PASS_strict`** = an answer must satisfy **all four**
(format compliance ∧ directional accuracy ∧ checkpoint validity ∧ anchor
faithfulness). A weighted sum is **deliberately not used**: a
well-formed, nicely anchored answer that points the wrong way is
*useless* — averaging would let format points mask a wrong direction.
The metrics are not interchangeable, so they are AND-ed.

Reported separately, un-weighted, for diagnosis:
- **per-metric pass rate** — which metric fails most,
- **directional accuracy** alone — the closed-loop rate, the number
  watched most closely,
- **hallucination rate** — how often anchor faithfulness fails,
- **median heading error δ** — continuous, for the `*-implicit` /
  `*-explicit` (compass-free) conditions.

**Where it runs.** Zero-shot baselines (`B-given`, `B-infer`) run
**locally** on the RTX 3060 — inference only, no training. The LoRA
conditions are trained on Modal and evaluated straight after each run.
PASS_strict is compared across the 6 conditions for **each** ablation.

*Prior round, for reference:* a compass-free explicit-CoT LoRA reached
52.9 % PASS / median heading error 20.8° (vs base 99°); the with-compass
ceiling was 100 %.

---

## 5. Visualizations

Standalone HTML in `viz/`:
1. **The 27 candidate POIs** with signature icons — `poi_candidates_map.html`
   (`python -m src.poi --map`). ✅ built.
2. POIs found per video frame, on a map (with the POI photo at its pin).
3. Bought Street View panos highlighted on the map.
4. Recovered video-frame GPS on the map, **overlaid with the POI region
   polygons (Q8)** — so each frame's area and POI assignment (§3) is
   visible at a glance.
5. The 8-video routes derived from the images (§2.8).
6. A Q&A viewer — photo + question + generated answer — to sanity-check
   the instruction tuning.

---

## 6. Repo structure & environment

```
navlm_v2/
  config.py        paths, bbox, thresholds, models (relative paths only)
  src/             pipeline modules — `python -m src.<module>`
  tests/           pytest unit tests
  reference/       old navlm_ss code, read-only
  logs/            daily logs + infra.md
  viz/             generated HTML
  videos/          source videos (gitignored)
  data/  →  DATA_ROOT (gitignored)
```

- Reuses `navlm_ss/.venv` (torch 2.5.1+cu124, transformers, modal,
  yt-dlp, imagehash, opencv, pytest). `ffmpeg` installed.
- Rules: relative paths only; no secrets in code; each stage runnable
  and unit-tested in isolation.
- Code → GitHub · datasets/checkpoints → Hugging Face · data on local disk.

---

## 7. Roadmap

1. ✅ Scaffold `config.py`, `src/`, `tests/`.
2. ✅ Stage 1–2: video acquisition + frame extraction with quality filter.
3. ✅ `src/poi.py` (27 candidates + map), `src/streetview.py` (grid
   crawl), `src/reconcile.py` (weighted score), `train_modal.py` (Modal
   LoRA). 24 pytest tests.
4. GPS recovery: DINOv2 embed/match + VLM place-naming → feed
   `src/reconcile.py` + heading; then HMM road-snapping.
5. Sample test + `MIN_SIM` / `RECONCILE_TAU` tuning before the full crawl.
6. Visualizations + 8-video route map.
7. Phase B: routing + Gemini-2.5-Pro annotation (5-sample trial first).
8. Phase C/D: `train_modal.py` LoRA run + local zero-shot + eval.
