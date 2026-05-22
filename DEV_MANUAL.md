# NavLM v2 — Developer Manual

**Status:** planning. Nothing here is executed yet. This document is the
design we agree on *before* writing code. Each stage has two parts —
**[v1]** how `navlm_ss` did it (with `file:line` refs into
`reference/`), and **[v2]** what we change and why.

---

## 0. What NavLM is

Fine-tune a small vision-language model to give **spoken walking
directions from a phone photo + GPS, without a compass**. The model
infers camera heading by triangulating visible landmarks against a
nearby-POI map. All training data is self-supervised from 8 unlabelled
YouTube Zurich walking-tour videos — no manual GPS labelling.

The contribution is the **self-supervised data pipeline**, not the model.

---

## 1. Why rebuild

Problems found in `navlm_ss` that v2 fixes:

| # | Problem | Fix in v2 |
|---|---------|-----------|
| 1 | DINOv2 visual matching: ~only a small fraction of matches are real; the rest are forced argmax matches on look-alike facades | similarity floor + independent VLM geo-check + better reference imagery (§5.5) |
| 2 | Mapillary reference index: low quality, coverage offset NE, missed the south | replace with Google Street View grid (§5.4) |
| 3 | No image-quality filter — blurry / junk frames pass extraction | add blur + exposure + duplicate filtering (§5.2) |
| 4 | Hardcoded Linux paths (`/pub/evaluation_group/...`), API keys in source | `config.py` + `DATA_ROOT` + `.env` (§9) |
| 5 | Train/test split only held out one video of the *same* landmarks | POI-based hold-out (§6) |
| 6 | Monolithic 12 numbered steps, stale docstrings | modular stages, this manual |

---

## 2. Scope of v2 (now)

**Phase A first** — raw video → trusted (GPS, heading) frames — because
that is where the matching is broken. Phases B–D (instruction synthesis,
training, eval) are designed here but built after Phase A is solid.

```
videos ─▶ frames ─▶ quality filter ─▶ POI scan (Gemma/Gemini)
                                          │
   Street View reference grid ◀───────────┘
        │
        ▼
   GPS recovery:  DINOv2 match  +  VLM geo-check  +  OCR landmarks
        │                    │
        ▼                    ▼
   OSM + HMM road-snapping ─▶ trusted (GPS, heading) frames   ◀── Phase A ends
        │
        ▼  Phase B
   route planning ─▶ VLM instruction annotation (Gemini Pro) ─▶ verify
        │
        ▼  Phase C / D
   LoRA SFT (Modal GPU)  ·  zero-shot baselines (local)  ·  eval
```

---

## 3. Stage-by-stage design

### 3.1 Video acquisition

**[v1]** `reference/toolbox/fetch_videos.py` — `yt-dlp` as a subprocess,
format `-f best[ext=mp4]/best` (no resolution cap), saved to
`data/cities/{city}/videos/%(id)s.mp4`. The 8 videos are enumerated in
`pipeline/config.py:VIDEOS` (1 main + 7 extra).

**[v2]** Keep `yt-dlp`. The 8 URLs are now recorded in
`milestone2/videos/video_urls.md`. Download once to local `DATA_ROOT`.
`saturday_morning` reserved as hold-out video.

### 3.2 Frame extraction — video → ~27k images

**[v1]** `reference/toolbox/extract_frames.py` `dedup_scene_change()`:
1. ffmpeg dense sample at **1 fps**, `-q:v 3`, into `<video>_dense/`
2. 64-bit perceptual hash (`imagehash.phash`) of every dense frame
3. greedy keep — first frame always; keep next only if pHash Hamming
   distance from last-kept ≥ **`PHASH_THRESHOLD=10`**
   (`<6` near-dup skip · `6–14` keep · `>14` very different)

Result: 27,075 frames across 8 videos. **No quality/blur filter** — only
pHash dedup. Blurry-but-distinct frames are kept.

**[v2] Add a quality filter** after dense sampling, before/with dedup:
- **Blur** — variance of Laplacian; drop frames below a threshold
  (motion blur from the walker is common in these videos)
- **Exposure** — drop near-black / blown-out frames (mean luma + clip %)
- **Duplicate** — keep the pHash dedup (`PHASH_THRESHOLD` ~10), applied
  *after* the quality gate so we don't keep a blurry frame as the
  "representative" of a scene
Order: dense 1 fps → blur+exposure gate → pHash dedup → keep set.
Log per-video keep/drop counts and reasons.

### 3.3 POI layer

**[v1] Three POI tables:**
- `landmarks_zurich_osm.json` — **453** POIs, auto-extracted via osmnx
  Overpass (`extract_osm_pois.py`), bbox **8.520–8.570 E, 47.360–47.395 N**
- `zurich_landmarks_gps.py` — **31** hand-curated landmarks (core 2 km)
- `scenery_pois.py` — **13** hand-written streets/river/lake/bridges
  with per-POI radius

**Gemma POI scan** (`scan_video_pois_multi.py`): `google/gemma-4-31b-it`
via local vLLM, one request per frame, every 20th frame, a fixed
**26-candidate** POI list in the prompt ("identify ALL landmarks clearly
visible … reply `POI: <name>`"). Output `_video_poi_multi.jsonl`:
`{video, frame_id, visible_pois[]}` — 1,358 rows, 25 distinct POIs found.

**[v2]**
- Keep the 3 tables; the **GPS scope** of the project = the OSM bbox
  above (~5.3 km × 3.9 km of central Zurich) — this defines where we buy
  Street View (§3.4).
- Re-run the POI scan on **all** quality-filtered frames (not every-20th)
  with **Gemini** instead of Gemma (no vLLM server needed; stronger).
- Keep the same output schema so indexing (§3.7) is unchanged.

### 3.4 Reference imagery — Google Street View

**[v1]** Mapillary — 5,000 images, low quality, coverage offset. Cause of
problem #2.

**[v2] Street View grid** (already prototyped — `reference/fetch_streetview_grid.py`):
- **What to buy:** panoramas on a grid over the POI GPS bbox (§3.3).
- **How many / where:** the *free* metadata endpoint scans a grid (every
  ~50 m) and returns every panorama ID + exact GPS — found **1,915
  panoramas** in the old-town box at $0. Then the **Street View Static
  API** ($7 / 1000 images) downloads 4 headings (N/E/S/W) per panorama.
- **The $5 trial we ran:** restricted to a 178-pano core sub-box → 712
  images ≈ $4.98, to validate the pipeline cheaply before the full spend.
- **Full crawl estimate:** 1,915 panos × 4 headings ≈ $54.
Every Street View image carries exact pano GPS + capture date — it is a
clean, gap-free, pedestrian-level replacement for Mapillary.

### 3.5 GPS recovery — the core fix

**[v1]** DINOv2 (`dinov2-base`, avg-pool, 768-d) embeds each video frame,
cosine-matches against the Mapillary index, takes the top-k, medians the
GPS. Confidence = top-k GPS dispersion only. **Failure:** cosine
similarity is *relative* — DINOv2 always returns an argmax, so on
repetitive facades it confidently matches the wrong place. Estimated
only a small fraction of matches are genuine.

**[v2] Three independent GPS hypotheses, then reconcile:**
1. **DINOv2 match** against the Street View index — *plus an absolute
   similarity floor* (`--min-sim`, already added to
   `reference/toolbox/visual_match_gps.py`): below-threshold matches are
   rejected, not argmax-forced.
2. **VLM geo-localization** (`reference/pipeline/step_13_vlm_geocheck.py`):
   a VLM independently reasons "where in Zurich is this?" → resolved to
   GPS via the POI tables.
3. **OCR landmarks** — sign text → `zurich_landmarks_gps.py`.

Reconcile: if the hypotheses **agree** (within a variance threshold),
accept; if they **disagree**, drop the frame. Then **OSM + HMM
road-snapping** (Newson-Krumm Viterbi over the osmnx walking graph)
smooths the accepted GPS sequence onto real walkable geometry.

> **Open decision D-C (§4):** is DINOv2 the primary hypothesis and VLM
> the check, or VLM primary and DINOv2 the check, or equal vote?

**Sample test to run first** (cheap, before committing): take the $5
Street View batch + a sample of quality-filtered video frames from the
same core area → DINOv2 match → eyeball + measure. Ground truth must
**not** be Mapillary-derived (that was the flaw in the earlier A/B test).

### 3.6 Routing

**[v1]** `reference/toolbox/way_planner.py` — osmnx + networkx; pickled
walking graph; `nx.shortest_path(weight="length")`. Absolute bearings →
relative actions via `_action_for`: `|Δ|≤35°` continue ahead · `|Δ|>135°`
turn around · else left/right by sign. `ACTION_DELTA = {continue:0,
left:-90, right:+90, around:180}` (used by the closed-loop verifier).

**[v2]** Keep this — it is geometric and correct. Only re-point paths.

### 3.7 Instruction-tuning annotation

**[v1]** `synth_unified.py` + `synth/prompts.py`: Gemma teacher
(`gemma-4-31b-it`) sees photo + GPS + heading + nearby POIs + route, and
emits `<thinking>` (6 labelled steps) + `<answer>` (2–4 TTS-friendly
sentences). Destinations sampled with tier weighting **0.7 / 0.25 / 0.05**
(iconic / mid / small). Verified by a format verifier + a directed visual
verifier + the closed-loop angular verifier (`δ<30°` strict).

**[v2]** Same structure, but **redo annotation with Gemini Pro** as the
teacher (stronger reasoning, no vLLM server). Re-use the v3 prompt; keep
the closed-loop verifier as the hard gate.

### 3.8 Image ↔ POI indexing

**[v1]** Two indexes, bridged by GPS:
- `_video_poi_multi.jsonl` — `{video, frame_id, visible_pois[]}` (video side)
- `landmark_visibility.jsonl` — `{frame_id, lat, lon, candidate_poi,
  distance_m, verifier, raw_response}` (Mapillary side)

**[v2]** Keep the schema. A video frame is traceable POI→images and
image→POIs. This also gives us the **per-POI dataset distribution** (§5).

---

## 4. Open design decisions — need your sign-off

| ID | Decision | Options | My recommendation |
|----|----------|---------|-------------------|
| D-A | v2 scope now | Phase A only / full A–D | **Phase A only**, design B–D |
| D-B | `reference/` old code | keep as reference / delete | **keep** (done) |
| D-C | GPS-recovery logic | DINOv2-primary+VLM-check / VLM-primary+DINOv2-check / equal vote | **equal vote** — 3 hypotheses, drop on disagreement (most robust) |
| D-D | reference imagery | Street View only / + Mapillary fallback | **Street View only** |
| D-E | train/test split | hold-out video only / + POI hold-out | **both** — hold out `saturday_morning` *and* a set of destination POIs |
| D-F | annotation teacher | Gemma / Gemini Pro / Claude | **Gemini Pro** (your call) |
| D-G | quality filters | which to apply | blur (Laplacian var) + exposure + pHash dedup |

---

## 5. Train / test split

**[v1]** Held out the whole `saturday_morning` video (255 samples). But
that video walks the *same* landmarks as training → only tests "new
footage of known places."

**[v2] Two-axis hold-out:**
- **Camera axis** — keep holding out the `saturday_morning` video
  (prevents temporal leakage).
- **Destination axis** — reserve a set of destination POIs that are
  *never used as a training destination*, so the test set routes to
  places unseen in training (true generalization).

Subtlety to decide: a held-out POI may still appear *in the background*
of training frames. Pick deliberately — "never a destination" vs "never
seen at all." Recommendation: "never a destination" (achievable; still a
strong test).

---

## 6. Experiments & training

**[v1]** 6 conditions (3 prompts × base/LoRA). Headline: compass-free
explicit-CoT LoRA (C3) reached 52.9% PASS / median heading error 20.8°
vs base 99°. Heading could not be *fully* removed (C3 still 47 pp below
the with-compass ceiling C1=100%).

**[v2] This round:**
- **Zero-shot baselines** → run **locally** on the RTX 3060 (small,
  no training).
- **LoRA training** ("the plannings") → run on **Modal** GPU.
- Re-evaluate on the new two-axis hold-out (§5).

---

## 7. Dataset analysis & visualization

To be produced once Phase A data exists:
- **Per-POI distribution** — frame count per POI from
  `_video_poi_multi.jsonl`; histogram (already prototyped logic).
- **Coverage map** — Leaflet map (`reference/make_streetview_map.py`
  pattern) overlaying: Street View reference panos, video-frame GPS
  estimates, and POIs — to see gaps and verify matches spatially.

---

## 8. Repo structure, config, environment

```
navlm_v2/
  DEV_MANUAL.md          this file
  README.md
  config.py              DATA_ROOT, bbox, thresholds, model names
  .env / .env.example    API keys (gitignored)
  .gitignore             .venv data results *.pdf __pycache__ .env
  logs/                  daily logs
  src/                   the pipeline (modular, from scratch)
  reference/             old navlm_ss code — read-only reference
  data/   -> local DATA_ROOT (gitignored; raw inputs read from navlm_ss)
```

- **Environment:** reuse `navlm_ss/.venv` (torch 2.5.1+cu124, transformers,
  torchvision). Add `imagehash`, `yt-dlp`, `osmnx`, `opencv` for the new
  filters. `ffmpeg` must be installed and on PATH.
- **Storage:** code → GitHub. Datasets + checkpoints → Hugging Face.
  Raw data on local disk; no secrets in git.

---

## 9. Roadmap

1. **Sign off this manual** (decisions D-A … D-G).
2. Scaffold `navlm_v2/src/` + `config.py` + `.env` + `.gitignore`; first
   git commit; push to GitHub.
3. Phase A: video → frames + quality filter → POI scan → Street View
   reference → GPS recovery → OSM/HMM → trusted frames.
4. Sample test (§3.5) before the full $54 Street View crawl.
5. Phase B: routing + Gemini-Pro instruction annotation.
6. Phase C/D: Modal LoRA training + local zero-shot + eval.
```
