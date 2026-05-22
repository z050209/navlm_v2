# NavLM v2 — Developer Manual

**Status:** planning. Nothing here is executed yet. Each stage has two
parts — **[v1]** how `navlm_ss` did it (with `file:line` refs into
`reference/`), and **[v2]** what we change and why.

> **2026-05-22** — expanded to answer the 13-point review in
> `experiment.sh`: OCR dropped (§3.5), video download (§3.1), frame
> extraction (§3.2), POI extraction + Gemma scan + 3-table relationship
> (§3.3), bbox margin (§3.4), min-sim / breaking criteria / heading
> (§3.5), GPS-photos-as-GT question (§3.6), Gemini-2.5-Pro annotation
> (§3.7), 8-video route map (§3.8), Gemini budget (§3.9), experiment
> settings (§6), visualizations (§7).

---

## 0. What NavLM is

Fine-tune a small VLM to give **spoken walking directions from a phone
photo + GPS, without a compass** — the model infers camera heading by
triangulating visible landmarks against a nearby-POI map. Training data
is self-supervised from 8 unlabelled YouTube Zurich walking-tour videos.
The contribution is the **self-supervised data pipeline**, not the model.

---

## 1. Why rebuild

| # | Problem in `navlm_ss` | Fix in v2 |
|---|----------------------|-----------|
| 1 | DINOv2 matching forces an argmax — most matches on look-alike facades are wrong | similarity floor + VLM geo-check + Street View index (§3.5) |
| 2 | Mapillary index: low quality, coverage offset, missed the south | Google Street View grid (§3.4) |
| 3 | No image-quality filter — blurry frames pass extraction | blur + exposure filter (§3.2) |
| 4 | Hardcoded Linux paths, API keys in source | `config.py` + `DATA_ROOT` + `.env` |
| 5 | Train/test split held out one video of the *same* landmarks | POI-based hold-out (§5) |
| 6 | Monolithic 12 numbered steps | modular `src/`, this manual |

---

## 2. Scope of v2 (now)

**Phase A first** — raw video → trusted (GPS, heading) frames.

```
videos ─▶ frames ─▶ quality filter ─▶ POI scan (Gemma, kept from v1)
                                          │
   Street View reference grid ◀───────────┘
        │
        ▼
   GPS recovery:  DINOv2 match  +  VLM geo-localization (Gemini Flash)
        │                    │
        ▼                    ▼
   OSM + HMM road-snapping ─▶ trusted (GPS, heading) frames   ◀── Phase A ends
        │
        ▼  Phase B
   route planning ─▶ instruction annotation (Gemini 2.5 Pro) ─▶ verify
        │
        ▼  Phase C / D
   LoRA SFT (Modal GPU)  ·  zero-shot baselines (local)  ·  eval
```

---

## 3. Stage-by-stage design

### 3.1 Video acquisition

**[v1]** `reference/toolbox/fetch_videos.py` — `yt-dlp` subprocess,
format `best[ext=mp4]/best`, saved to `data/cities/{city}/videos/`.

**[v2]** The 8 URLs are in `milestone2/videos/video_urls.md`;
`saturday_morning` is the evaluation hold-out.

**Download — code & command:**

```bash
pip install yt-dlp                       # one-time

IDS="h7saB68KE5M g21yfR4yNd8 F8KpE5iEvW0 8zcXNiWRgtA \
     3BnA_kP2HHY JUuggKe733s 5175ziTF3Gc QU1HxFTuqPY"

for id in $IDS; do
  yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/best[ext=mp4]" \
         -o "<DATA_ROOT>/videos/%(id)s.mp4" \
         "https://www.youtube.com/watch?v=$id"
done
```

~49 GB total.

### 3.2 Frame extraction — video → ~27k images

**[v1]** `reference/toolbox/extract_frames.py` `dedup_scene_change()`:
ffmpeg dense-sample at 1 fps (`-q:v 3`) → 64-bit pHash of each frame →
greedy keep if pHash Hamming distance from last-kept ≥ `PHASH_THRESHOLD=10`.
Result 27,075 frames. **No quality/blur filter** — blurry frames pass.

**[v2] Add a quality filter:** dense 1 fps → **blur** (variance-of-Laplacian
threshold) + **exposure** (mean-luma / clip-% gate) → pHash dedup → keep
set. Log per-video keep/drop counts.

**Running it — code & command:**

```bash
python reference/toolbox/extract_frames.py --city zurich --dedup \
    --dense-fps 1.0 --phash-threshold 10
```

Prerequisites — **none set up on this machine yet**:
- `ffmpeg` installed + on PATH
- `pip install imagehash pillow`
- videos downloaded (§3.1)
- the script's hardcoded `ROOT = /pub/evaluation_group/...` repointed to
  the local `DATA_ROOT`

One-video sanity check with ffmpeg alone:
```bash
ffmpeg -i <video>.mp4 -vf fps=1 -q:v 3 out/dense_%06d.jpg
```
v2 ships a path-portable extractor in `src/` with the blur filter built in.

### 3.3 POI layer

**Three POI tables — relative paths and purpose:**

| File | Relative path | Rows | Purpose |
|------|---------------|------|---------|
| `landmarks_zurich_osm.json` | `data/cities/zurich/landmarks_zurich_osm.json` | 453 | OSM auto-extracted point POIs — breadth |
| `zurich_landmarks_gps.py` | `toolbox/zurich_landmarks_gps.py` | 31 | hand-verified core landmarks — precision |
| `scenery_pois.py` | `toolbox/scenery_pois.py` | 13 | streets / river / lake / bridges (OSM *ways*, not points) |

**How `landmarks_zurich_osm.json` is extracted** — `extract_osm_pois.py`
queries OpenStreetMap via osmnx Overpass (`ox.features_from_bbox`):

```bash
python reference/toolbox/extract_osm_pois.py \
    --bbox 8.520,47.360,8.570,47.395   # W,S,E,N
```
It keeps 7 tag groups (tourism, historic, amenity, public_transport/railway
station, leisure, place), filters names (3–30 chars, uppercase start,
blocklist), and writes `{lat, lon, aliases, kinds, kind_label}` per POI →
**453** entries.

**How the bbox was determined:** `8.520,47.360,8.570,47.395` is central
Zurich's old town — the area the walking tours cover (Hauptbahnhof →
Altstadt → Grossmünster → Bellevue / lakefront). ≈ 3.8 km E–W × 3.9 km
N–S. This bbox defines the **GPS scope of the whole project**.

**Purpose of `zurich_landmarks_gps.py`** — 31 hand-curated landmark
name→(lat, lon, aliases), manually verified from OSM. v1 used it for
OCR sign-text → GPS. **v2 (OCR removed):** it is the high-precision
table the VLM geo-localizer uses to resolve a named place → GPS, and a
routing destination table.

**Purpose of `scenery_pois.py`** — 13 hand-written entries for features
OSM tags as *ways/polygons* (streets, the Limmat, the lake, bridges) that
point-node extraction misses. Each carries a custom `radius_m` (streets
~300 m, lake 600 m, bridges ~80 m) so proximity checks use a
feature-appropriate radius instead of a fixed 50 m.

**Relationship of the 3 tables** — complementary, merged into one POI DB:
OSM (breadth, 453 points) + curated (precision on 31 core landmarks, with
OCR-era aliases) + scenery (13 non-point areas with radii). v1's
`step_10` `load_poi_db()` merges OSM + scenery; synth merges all three.

**Gemma POI scan** — `reference/toolbox/scan_video_pois_multi.py`:
- **Model:** `google/gemma-4-31b-it` via local vLLM, one request per
  frame, every 20th frame.
- **The fixed 26-candidate POI list** is hardcoded as `CANDIDATE_POIS`
  in `scan_video_pois_multi.py:24-36` — a hand-picked short list of the
  most iconic Zurich landmarks + scenery (Hauptbahnhof, Lindenhof,
  Paradeplatz, Fraumünster, Grossmünster, …, Bahnhofstrasse, Limmat,
  Lake Zurich). It is a *subset* of the 453, kept short so Gemma picks
  from a manageable set.
- **Input:** one frame image + a prompt embedding the 26 candidates
  ("identify ALL landmarks clearly visible … reply `POI: <name>`, or
  `POI: none`").
- **Output:** parsed to `_video_poi_multi.jsonl`,
  `{video, frame_id, visible_pois[]}` — 1,358 rows, 25 distinct POIs.

**[v2] Keep the existing Gemma POI scan output as-is — do NOT rerun with
Gemini.** The 3 tables and `_video_poi_multi.jsonl` carry over unchanged.

### 3.4 Reference imagery — Google Street View

**[v1]** Mapillary — 5,000 images, low quality, coverage offset (problem #2).

**[v2] Street View grid** (`reference/fetch_streetview_grid.py`): the free
metadata endpoint scans a grid and returns every panorama ID + exact GPS;
the Street View Static API ($7/1000) then downloads 4 headings per pano.
The $5 trial (178 panos → 712 imgs) is done; full crawl ≈ $54.

**[v2 — bbox margin (your Q6).** Yes — the Street View bbox should be
**larger than the POI bbox**. A POI sitting on the edge, or a route
segment that leaves the POI box, still needs reference imagery. Plan:
crawl bbox = POI bbox + **~300 m margin** on each side ≈
`8.515, 47.355, 8.575, 47.400`. Cheap (metadata scan is free) and removes
edge blind-spots.

### 3.5 GPS recovery — the core fix

**[v1]** DINOv2 (`dinov2-base`, avg-pool) embeds each frame, cosine-matches
the Mapillary index, medians the top-k GPS. Cosine similarity is
*relative* → argmax always returns *something* → wrong matches on
repetitive facades.

**[v2] Two independent GPS hypotheses, then reconcile:**
1. **DINOv2 match** vs the Street View index, **with an absolute
   similarity floor** `--min-sim` (in `reference/toolbox/visual_match_gps.py`).
2. **VLM geo-localization** — `reference/pipeline/step_13_vlm_geocheck.py`,
   run with **Gemini Flash** (cheap). Input: a frame + the match's GPS;
   output JSON `{landmark, lat, lon, confidence, reasoning}` →
   `{verdict, variance_m, vlm_gps, …}`.

> **OCR-landmark recovery dropped** — v1's `ocr_paddle.py` +
> `landmark_match.py` removed; brittle, and a 3rd path adds little.

**`--min-sim` value (your Q7):** not fixed up front — **tune it**. Run
matching at several `--min-sim` values, pick the one that keeps the most
frames while quality stays high, **sanity-checking 10% of matches** by
eye. Run the **$5 sample-test batch first** to calibrate cheaply.

**Breaking / reject criteria:**
- DINOv2: best cosine `< min_sim` → reject (no GPS from vision).
- Reconcile: `variance_m = haversine(dino_gps, vlm_gps)`; if
  `variance_m > threshold` (≈150 m) → **drop the frame**; if they agree → accept.
- VLM: `confidence=low` / unparseable → `GEO_UNKNOWN`, not a hard drop.

**How heading is concluded:**
- *v1* (`compute_frame_heading.py`): take the `compass_angle` of the
  top-k matched **Mapillary** images, outlier-filter (>90° from circular
  median), circular-mean → heading; confidence by circular std (<20°
  high, 20–45° medium).
- *v2:* each Street View crop is rendered at a **known heading**
  (0/90/180/270°). The matched crop's heading *is* the frame's heading
  estimate — average over the top-k matched crops. No separate
  Mapillary-compass step needed.

Then **OSM + HMM road-snapping** (Newson-Krumm Viterbi over the osmnx
walking graph) smooths the accepted GPS onto walkable geometry.

### 3.6 Routing

**[v1]** `reference/toolbox/way_planner.py` — osmnx + networkx, pickled
walking graph, `nx.shortest_path(weight="length")`. Bearings → relative
actions: `|Δ|≤35°` continue · `|Δ|>135°` turn around · else left/right.
`ACTION_DELTA = {continue:0, left:-90, right:+90, around:180}`.

**[v2]** Keep as-is — geometric and correct. Only re-point paths.

**Your Q8 — can Street View ("GPS") photos be GT for instruction
annotation?** Yes, in two distinct ways:
1. **As the GPS/heading ground truth** — every Street View image has
   *exact* pano GPS + heading. Video frames inherit GPS by matching
   against them (§3.5). ✅ This is the core use.
2. **As extra annotation inputs** — you *could* annotate Street View
   images directly: they carry exact GPS+heading, so no recovery error,
   giving very clean instruction-tuning samples. ⚠️ Caveats: (a) a domain
   gap — Street View ≠ phone-video, the deployment distribution; (b)
   Google Maps ToS restricts training on Street View imagery. Suggested:
   use them as **GT for GPS/heading**, keep training *inputs* = video
   frames; optionally add a small Street View slice as an ablation.

### 3.7 Instruction-tuning annotation

**[v1]** `synth_unified.py` + `synth/prompts.py`: Gemma teacher sees
photo + GPS + heading + nearby POIs + route → `<thinking>` (6 steps) +
`<answer>` (2–4 TTS sentences). Destinations tier-weighted 0.7/0.25/0.05.
Verified by format + visual + closed-loop (`δ<30°`) verifiers.

**[v2 — Gemini 2.5 Pro teacher (your Q9).** Same prompt structure, swap
the teacher to **`gemini-2.5-pro`** via `backends.call_gemini`. Keep the
closed-loop verifier as the hard gate. **Run 5 samples first** — the v2
`src/` annotation module will take a `--limit 5` flag; we inspect the 5
outputs (thinking + answer + verifier verdict) before the full run.
Code: `synth_unified.py` logic + `call_gemini(model="gemini-2.5-pro")`;
shown for review before execution. Budget in §3.9.

### 3.8 Image ↔ POI indexing & route map

**[v1]** Two GPS-bridged indexes — `_video_poi_multi.jsonl`
(`{video, frame_id, visible_pois[]}`) and `landmark_visibility.jsonl`
(`{frame_id, lat, lon, candidate_poi, distance_m, verifier, …}`).

**[v2 — keep the schema (your Q10).** Once frames are mapped to GPS,
plot a **route map for the 8 videos**: each video's recovered GPS
sequence as a coloured polyline on one Leaflet/folium map (v1 had
`visualize_paths.py`). This both visualizes the 8 walks and is a sanity
check on the GPS recovery.

### 3.9 Cost — Gemini budget calculation

Per 1M tokens: **Gemini 2.5 Pro** $1.25 in / $10 out · **Flash**
$0.30 / $2.50. Estimates assume ~560 input tokens/image; ±50%.

| Workload | Calls | in/call | out/call | **Pro** | Flash |
|----------|------:|--------:|---------:|--------:|------:|
| POI scan | — | — | — | — | *kept from v1, not rerun* |
| VLM geo-check — candidate frames | ~5,000 | ~810 | ~250 | ~$18 | **~$4** |
| Annotation — 5 dest/frame | ~11,000 | ~1,960 | ~900 | **~$126** | ~$31 |
| Annotation — 3 dest/frame | ~6,600 | ~1,960 | ~900 | **~$76** | ~$19 |

**Reality check.** Annotation on Gemini Pro is the big cost; plus Street
View ~$54 — both draw on the same **$50 GCP credit**, which is **not
enough**. Plan: geo-check → **Flash** (~$4); annotation → **Pro** but at
3 dest/frame (~$76) or fewer; Street View → route-based crawl, not the
full $54 grid. $50 stays tight — flag for the budget owner. Track live
with `reference/track_spend.py`.

---

## 4. Open design decisions

| ID | Decision | Resolution |
|----|----------|-----------|
| D-A | v2 scope | Phase A first, design B–D |
| D-B | old code | kept in `reference/` ✅ |
| D-C | GPS-recovery logic | **DINOv2 + VLM equal vote** (OCR dropped); drop frame on disagreement |
| D-D | reference imagery | Street View only |
| D-E | train/test split | hold out `saturday_morning` video **+** a set of destination POIs (§5) |
| D-F | annotation teacher | **Gemini 2.5 Pro**; geo-check uses Gemini Flash |
| D-G | quality filters | blur (Laplacian var) + exposure + pHash dedup |

---

## 5. Train / test split

**[v2] Two-axis hold-out:**
- **Camera axis** — hold out the whole `saturday_morning` video (no
  temporal leakage).
- **Destination axis** — reserve a set of destination POIs *never used
  as a training destination*, so the test set routes to unseen places.

Decision: "never a destination" (achievable; still a strong test) — a
held-out POI may still appear in the background of training frames.

---

## 6. Experiments & training

**[v1] experiment settings** (`results/EXPERIMENT_REPORT.md`) — thesis:
*can a VLM learn to derive its own heading from the photo?* **6
conditions** = 3 prompt variants × {base, LoRA}:

| ID | model | prompt | heading given? |
|----|-------|--------|----------------|
| A1 | base Qwen2.5-VL-7B | v3 | yes |
| A2 | base | v4a (no heading, implicit CoT) | no |
| A3 | base | v4b (no heading, explicit CoT) | no |
| C1 | + LoRA | v3 | yes |
| C2 | + LoRA | v4a | no |
| C3 | + LoRA | v4b | no |

- **Training data:** 4,434 shared samples; v4a/v4b derived from v3 by
  regex (no teacher re-prompt). **Hold-out:** 255 samples from
  `saturday_morning`, excluded from training.
- **LoRA:** Qwen2.5-VL-7B, r=16, α=32, 4-bit NF4 base, BF16 adapters,
  2 epochs, lr 2e-4, batch 1 × grad-accum 8, images capped 448².
- **Eval:** 5 gates (format / sentence-count / closed-loop angle /
  checkpoint / anchor); headline metric PASS_strict.
- **Result:** compass-free explicit-CoT LoRA (C3) = 52.9% PASS, median
  heading error 20.8° (vs base 99°); with-compass ceiling C1 = 100%.

**[v2] this round:** zero-shot baselines **locally** (RTX 3060); LoRA
training on **Modal** A100 (§ infra.md §10); re-evaluate on the new
two-axis hold-out (§5).

---

## 7. Visualizations (your Q13)

Five artifacts, produced as data becomes available:

1. **Gemma POI map** — POIs found per video frame plotted on a map; if
   feasible, show the POI's photo at its pin.
2. **Street View coverage** — the bought Street View panos highlighted
   on the map (done once for the $5 batch — `make_streetview_map.py`).
3. **Mapped video frames** — after GPS recovery, highlight each video
   frame at its recovered GPS.
4. **Routes** — the route derived from the images (overlaps §3.8's
   8-video route map).
5. **Q&A viewer** — render each instruction-tuning sample (photo +
   question + generated answer) for a human sanity check.

All as standalone Leaflet/HTML in a `viz/` output folder.

---

## 8. Repo structure, config, environment

```
navlm_v2/
  DEV_MANUAL.md            this file
  config.py                DATA_ROOT, bbox, thresholds, model names
  .env / .env.example      API keys (gitignored)
  logs/                    daily logs + infra.md
  src/                     the v2 pipeline — modular, relative paths only
  reference/               old navlm_ss code — read-only
  viz/                     generated HTML visualizations
  data/  →  local DATA_ROOT (gitignored)
```

- **Environment:** reuse `navlm_ss/.venv` (torch 2.5.1+cu124,
  transformers, modal). Add `imagehash, yt-dlp, osmnx, opencv`. `ffmpeg`
  must be installed.
- **Rules:** relative paths only (no hardcoded absolute paths); no
  secrets in code; every stage independently runnable.
- **Storage:** code → GitHub · datasets/checkpoints → Hugging Face ·
  raw data on local disk.

---

## 9. Roadmap

1. Scaffold `src/` + `config.py` + `.env.example`.
2. Phase A: download videos → extract+filter frames → (keep Gemma POI
   scan) → Street View crawl → GPS recovery → OSM/HMM → trusted frames.
3. Sample test (§3.5) + `--min-sim` tuning before the full crawl.
4. Visualizations (§7) + 8-video route map (§3.8).
5. Phase B: routing + Gemini-2.5-Pro annotation — **5-sample trial first**.
6. Phase C/D: Modal LoRA training + local zero-shot + eval.
