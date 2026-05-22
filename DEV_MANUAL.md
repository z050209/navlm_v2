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
chosen as a scene's representative. Per-video keep/drop counts are logged
to `frames/extract_report.json`.

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
- **Scenery entries** — 13 hand-added entries for features OSM tags as
  *ways/polygons*, which point extraction misses: streets (Bahnhofstrasse,
  Niederdorfstrasse…), the Limmat, Lake Zurich, bridges. Each carries a
  custom `radius_m` (streets ~300 m, lake 600 m, bridges ~80 m) so
  proximity checks use a feature-appropriate radius. These matter for
  navigation ("walk along Bahnhofstrasse") and are folded into the table.

> **Dropped (Q1):** the separate 31-entry `zurich_landmarks_gps.py`
> hand-curated table. Its only unique role was OCR alias matching; OCR is
> removed (§2.5), and the OSM 453 already covers those landmarks. The
> scenery entries are kept because they cover way/area features OSM
> point-extraction genuinely misses.

**POI scan of the video frames** — `scan_video_pois_multi.py` had a VLM
(Gemma) look at frames and list visible POIs from a fixed **26-candidate
list**. That list (`CANDIDATE_POIS`) is a hand-picked shortlist of the
most iconic Zurich landmarks + scenery — effectively the **tier-1 /
"L1" iconic set** (Hauptbahnhof, Lindenhof, Paradeplatz, Fraumünster,
Grossmünster, Bahnhofstrasse, the Limmat, Lake Zurich, …). It is short
on purpose, so the VLM picks from a manageable menu. **The existing scan
output (`_video_poi_multi.jsonl`) is kept as-is — not rerun.**

### 2.4 Street View reference grid

The reference index against which video frames are matched is Google
Street View panoramas (`reference/fetch_streetview_grid.py`):

- The **free metadata endpoint** scans a grid and returns every panorama
  ID + exact GPS at $0.
- The **Street View Static API** ($7/1000) then downloads 4 headings
  (N/E/S/W) per panorama.

**Crawl bbox (Q3).** The grid should cover **where the videos actually
walk**, not an arbitrary box. The route extent is bootstrapped from the
GPS bounding box of the POIs the video POI-scan found (§2.3), plus a
**~300 m margin** so a POI on the edge — or a route segment that leaves
the box — still has reference imagery. The metadata scan is free and
incremental, so the box can be expanded later if GPS recovery shows
routes near an edge.

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

**Heading (Q5).** Each Street View crop was rendered at a *known*
heading (0/90/180/270°). The frame's camera heading ≈ the heading of the
matched crop; averaging the headings of the top-k matched crops
(circular mean, outlier-filtered — the geometry is in
`reference/toolbox/compute_frame_heading.py`) gives a heading estimate
plus a spread-based confidence.

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

The teacher VLM (**Gemini 2.5 Pro**, Q6) is given a frame + its GPS +
heading + nearby POIs + the planned route, and produces `<thinking>`
(6 labelled reasoning steps) + `<answer>` (2–4 TTS-friendly sentences,
relative verbs anchored to a visible object). **3 destinations per
frame** (Q6), tier-weighted toward iconic POIs.

Every sample is gated by a **closed-loop verifier**: parse the action
verb from the answer, check `|heading + ACTION_DELTA[verb] −
route_bearing| < 30°`. Samples that fail are dropped.

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
   held-out video (`saturday_morning`). Tests new footage of (possibly)
   known places.
2. **POI generalization** — train on one set of destination POIs, test
   on a disjoint set. Tests routing to places never seen as a training
   destination.

---

## 4. Experiments & training

Prior result for reference: 6 conditions (3 prompt variants × {base,
LoRA}); the compass-free explicit-CoT LoRA reached 52.9 % PASS / median
heading error 20.8° (vs base 99°); the with-compass ceiling was 100 %.

This round: zero-shot baselines run **locally** (RTX 3060); LoRA training
on **Modal** A100 (`logs/infra.md §10`); evaluate under both ablations
(§3). LoRA config: Qwen2.5-VL-7B, r=16, α=32, 4-bit NF4 base, 2 epochs,
lr 2e-4.

---

## 5. Visualizations

Standalone HTML in `viz/`:
1. POIs found per video frame, on a map (with the POI photo at its pin).
2. Bought Street View panos highlighted on the map.
3. Recovered video-frame GPS highlighted on the map.
4. The 8-video routes derived from the images (§2.8).
5. A Q&A viewer — photo + question + generated answer — to sanity-check
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
3. Street View crawl module (`src/streetview.py`).
4. GPS recovery: DINOv2 embed/match + VLM place-naming + weighted score
   + heading + HMM snapping.
5. Sample test + `min_sim`/`τ` tuning before the full crawl.
6. Visualizations + 8-video route map.
7. Phase B: routing + Gemini-2.5-Pro annotation (5-sample trial first).
8. Phase C/D: Modal LoRA training + local zero-shot + eval (both ablations).
