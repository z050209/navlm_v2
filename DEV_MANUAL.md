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
videos ─▶ frames ─▶ quality filter ─▶ POI scan (Gemini Pro · Vertex AI)
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

Once per terminal, `cd` into the repo and activate the local conda env
`navlm_v2` (Python 3.11, on the C: drive — the earlier Drive venv hung
because every Python import had to read `.py` files from Google Drive):

```powershell
cd C:\Users\z0502\Desktop\cs231n\navlm_v2
conda activate navlm_v2          # or use the env python directly:
# function python { & "C:\Users\z0502\anaconda3\envs\navlm_v2\python.exe" @args }
```

The env was created via `conda create -n navlm_v2 -c conda-forge
python=3.11 pip -y`, then `pip install` of: torch (CUDA 12.4),
transformers, Pillow, numpy, tqdm, requests, folium, opencv-python,
imagehash, osmnx, networkx, modal, yt-dlp, pytest. Local SSD only —
no Google Drive in the import path.

Then run these one by one. Only ✅ steps exist yet; ⏳ land as built (§7).

**Pipeline run order** — Phase A is steps 1–8; later steps depend on
earlier ones (a step's *needs* are noted).

| # | Step | Command | Needs | Status |
|---|------|---------|-------|--------|
| 0 | sanity-check config | `python config.py` | — | ✅ |
| 0b| run the unit tests | `python -m pytest tests/ -q` | — | ✅ |
| 1 | build the OSM POI table | `python -m src.pois` | — | ✅ |
| 2 | (download videos — already present) | `python -m src.download_videos` | — | ✅ |
| 3 | extract frames (quality-filtered) | `python -m src.extract_frames` | step 2 | ✅ |
| 4 | POI scan — Gemini Pro (Vertex) → OSM match | `python -m src.poi_scan --limit 3` then `--every-n 30` | steps 1, 3 | ✅ |
| 4b| POI-scan map (matched POIs + derived bboxes) | `python -m src.viz_scan` | step 4 | ✅ |
| 5 | Street View — bbox / scan / download (targeted, §2.4) | `python -m src.streetview --bbox` · `--scan` · `--download` | step 4 | ✅ |
| 5b| **DINOv2 match pilot** — v2 frames vs the v1 712 SV images | `python -m src.dinov2_match --every-n 40 --min-sim 0.60` | step 3 + local copy of SV images | ✅ |
| 6 | **GPS recovery** (strict F1/F2/F3 + same-pano heading) | `python -m src.gps_recovery` | steps 3, 4, 5 | ✅ |
| 6b| GPS-recovery map + per-frame photo grid (sanity check) | `python -m src.viz_recovery` · `python -m src.viz_recovery_grid` | step 6 | ✅ |
| 7 | OSM + HMM road-snapping (snap GPS, correct heading) | `python -m src.road_snap` | step 6 | ⏳ |
| 8 | other visualizations | `python -m src.poi --map` · `python -m src.viz` | varies | ✅/⏳ |
| 9 | LoRA training on Modal | `modal run train_modal.py` | annotated data | ✅ |

(`--list` previews most steps; `--limit N` does a small trial run.)

Notes:
- Step 3 writes frames to `data/cities/zurich/frames/<name>/` plus an
  `extract_report.json` with per-video keep/drop counts.
- `config.py` locates `ffmpeg` automatically (installed via winget).
- Each `src` module also accepts `-h` for its options.

---

## 2. Pipeline stages

Each stage is one `src/` module, runnable on its own (see the run table
above). Every stage section gives **Code · In · Out · Run**.

**Phase A — video → trusted (GPS, heading) frames:**

| § | Stage | Module | Produces |
|---|-------|--------|----------|
| 2.1 | Video acquisition | `download_videos.py` | the 8 source `.mp4` |
| 2.2 | Frame extraction | `extract_frames.py` | quality-filtered frames |
| 2.3 | POI layer | `pois.py`, `poi_scan.py` | `pois.json`, `poi_scan.jsonl` |
| 2.4 | Street View reference | `streetview.py` | the SV image index |
| 2.5 | GPS recovery | `gps_recovery.py`, `reconcile.py` | per-frame GPS + heading |
| 2.6 | Routing | `routing.py`, `road_snap.py` | OSM routes, snapped GPS |

**Phase B and supporting:**

| § | Stage | Module |
|---|-------|--------|
| 2.7 | Instruction-tuning annotation (Gemini Pro) | `annotate.py` |
| 2.8 | Image ↔ POI indexing & route map | `viz.py` |
| 2.9 | Gemini API budget | — |

### 2.1 Video acquisition

8 YouTube walking-tour videos (`milestone2/videos/video_urls.md`);
`saturday_morning` is the evaluation hold-out.

**Download — `src/download_videos.py`** (a yt-dlp wrapper). It reads the
8 video IDs from `config.VIDEOS`, downloads each (best mp4 video + m4a
audio, ffmpeg-merged) and is resumable — it skips any file already
present.

**Code:** `src/download_videos.py` · **In:** `config.VIDEOS` (8 IDs) ·
**Out:** `videos/<name>.mp4` · **Run:**

```bash
python -m src.download_videos              # all 8
python -m src.download_videos --only saturday_morning
python -m src.download_videos --list       # list, do not download
```

All 8 videos are already in `videos/Zurich/` — this is for
reproducibility / a missing video only.

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

- **OSM-extracted points** — `src/pois.py` queries OpenStreetMap via
  osmnx Overpass over the project bbox and keeps point landmarks tagged
  tourism / historic / amenity / station / leisure / place, filtered by
  name. The bbox (`config.POI_BBOX` = 8.520, 47.360, 8.570, 47.395
  W,S,E,N) is central Zurich's old town — Hauptbahnhof → Altstadt →
  Grossmünster → lakefront, ≈ 3.8 km × 3.9 km. **This bbox is the GPS
  scope of the project.**
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

Each row → `{name, aliases, kind_group, osm_kind, kind_label,
description, lat, lon, geometry}` — `osm_kind` is the raw OSM tag
(`amenity=theatre`), `kind_label` a human descriptor derived from it
("a theatre", always present), `description` the OSM `description` tag
where it exists (≈ 3 % of POIs). Point POIs keep a lat/lon, ways/areas
keep the polyline/polygon. Output: `data/cities/zurich/pois.json` (1,289
POIs). Run: `python -m src.pois`.

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

**POI scan of the video frames — `src/poi_scan.py`.** This identifies
which named places each frame actually shows. It matters because **the
POIs that appear in the videos shape what the instruction tuning can
cover — i.e. what a user can ask the model to navigate to**, so the scan
must be honest about provenance.

- **Inference-based, open-set.** Each frame goes to **Gemini 2.5 Pro**
  (`config.GEMINI_SCAN`, via Vertex AI — see *API backend* below), asked
  to *reason about where the photo was taken* — from shop names,
  architecture, trams, churches,
  the lake — with **no** need for a visible street sign. It replies as
  JSON: `visible` (places directly seen / read), `guess` (its best
  inference of the street / square / area), `confidence`
  (high/medium/low) and a one-sentence `reasoning`. (This replaces both
  the v1 closed 27-candidate Gemma scan and an earlier strict "name only
  what is clearly visible" prompt that returned "none" for generic
  streetscapes.)
- **Matched to OSM.** Each `visible` name and the `guess` are resolved
  against the OSM POI table with the alias-aware, diacritic-folded
  `resolve_poi()` (Gemini gives a German + English variant so at least
  one form matches). A match keeps a `source` field (`visible` /
  `guess`); names matching nothing go to `unmatched` — surfacing real
  places missing from the OSM table.
- **Tiered by OSM tag.** Each matched POI is tagged **L1 / L2 / L3** by
  `poi_tier()` in `src/poi_scan.py`, classifying on the POI's **OSM tag**
  — not hand keywords. `src/pois.py:osm_kind()` records each POI's
  primary tag (`tourism=museum`, `amenity=place_of_worship`,
  `highway=primary`, …) into `pois.json`; `poi_tier()` maps that tag via
  the editable `TIER_BY_TAG` / `TIER_BY_KEY` tables. L1 = landmark
  categories you navigate *to* (attractions, churches, monuments,
  stations); L2 = supporting POIs (museums, parks, bridges, named
  streets, the river/lake); L3 = the rest. Current Zurich table:
  **L1 ≈ 115, L2 ≈ 1168, L3 ≈ 6** of 1,289 POIs.
  > Note: OSM tags give the *category*, not *fame* — every named street
  > lands in L2, so L2 is broad (≈ 770 streets). To tighten it, demote
  > `highway` in `TIER_BY_KEY`, or rank prominence by a separate signal
  > (e.g. POI-scan appearance frequency).
- **Output** `data/cities/zurich/poi_scan.jsonl`, per frame:
  `{video, frame_id, guess, confidence, reasoning, visible[],
  matched[{variants, matched_name, source, osm_name, osm_kind,
  kind_label, tier}], unmatched[]}`. This *is* the POI provenance — it
  shows where each frame is, and which POIs (and tiers) the dataset can
  anchor and route to.
- **Every API call is logged.** `src/gemini_api.py` owns the Gemini call
  and appends one line per call to `logs/gemini_api.jsonl` — input /
  output token counts, USD cost, `finishReason` and the **full response
  text** (the inspectable conversation log). It retries 429s and
  refreshes the Vertex OAuth token on 401.

**API backend (`config.GEMINI_BACKEND`).** Gemini 2.5 **Pro is not on
the Gemini-API free tier** (`limit: 0`), and an *Education* / free-trial
GCP billing account does **not** unlock the API-key paid tier. So the
scan reaches Pro through **Vertex AI** (`GEMINI_BACKEND = "vertex"`) —
OAuth via `gcloud`, billed to project `cs231n-navlm-2026`, which the
Education credit covers. `GEMINI_BACKEND = "aistudio"` uses the
`GEMINI_API_KEY` endpoint instead (free tier — Flash only).

**Run:**
```bash
python -m src.poi_scan --limit 3       # 3-frame trial first
python -m src.poi_scan --every-n 30    # the full run used here
python -m src.poi_scan                 # every 10th frame (default)
```

**`--every-n` (default 10)** is the temporal-stride control: the scan
processes every Nth of the 26,034 kept frames. The kept frames are
*already* pHash-deduped at extraction, so even `--every-n 1` has no
near-duplicates; `--every-n` simply trades coverage for cost —
- `10` (default) → ~2,600 frames;
- `30` → ~870 frames — the setting used (~3.5 h, ~$12 on Pro via Vertex);
- `1`  → all 26,034 — fullest, most API calls.
`--limit N` caps the total (for trial runs).

The `src/poi.py` 27-candidate list is now only the **iconic-POI
reference / map** (`--map`) — it is no longer the scan input.

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

The reference index that video frames are matched against is Google
Street View panoramas, built by **`src/streetview.py`**:

- **`--scan`** — the free metadata endpoint sweeps a grid and returns
  every panorama ID + exact GPS, at $0 → `panos.jsonl`.
- **`--download`** — the Street View Static API ($7/1000) then downloads
  4 headings (N/E/S/W) per panorama → `images/` + `meta.jsonl`.

**Crawl bbox (Q3).** The grid bbox is *derived*, not hand-set — it is
the **bounding box of the POIs the videos actually visit**,
**+ a ~300 m margin** (`config.SV_MARGIN_M`) so an edge POI, or a route
segment leaving the box, still has reference imagery:

```
bbox = (min_lon − m, min_lat − m, max_lon + m, max_lat + m)
       over the visited POIs,  m ≈ 300 m
```

`streetview.bbox_from_scan()` reads the `matched` POIs from
`poi_scan.jsonl` (§2.3), looks their GPS up in `pois.json`, and applies
the margin. Before the scan exists it falls back to the `src/poi.py` 27
candidates. The metadata sweep is free, so the box can be widened and
re-crawled incrementally if routes reach an edge.

**Code:** `src/streetview.py` · **In:** `poi_scan.jsonl` + `pois.json`
(→ bbox) + Google Maps key · **Out:**
`data/cities/streetview/zurich/{images/, panos.jsonl, meta.jsonl}` ·
**Run:** `python -m src.streetview --bbox` (print the derived box) →
`--scan` (free) → `--download` (Static API).

> **Targeted-crawl strategy — only buy where the videos go.** The free
> `--scan` finds ~1,915 unique panos in the central-Zurich bbox; at
> $7/1000 × 4 headings that's $54 for the lot. But the POI scan
> (`poi_scan.jsonl`) already tells us *where the 8 videos actually
> walked* (227 matched OSM POIs). A pano is only useful if it's near
> one of those — everywhere else in the bbox is dead weight.
>
> The recipe:
> 1. take each of the 227 matched POIs from `poi_scan.jsonl` (street
>    polylines, square polygons, point landmarks);
> 2. **buffer by ~150 m** (covers one city-block around each visited
>    POI);
> 3. union the buffers → the "visited footprint";
> 4. filter the 1,915 metadata-scanned panos → keep only those inside
>    the footprint (typically ~800–1,000 panos);
> 5. `--download` just that subset (~$22–28).
>
> | strategy | panos | cost | what it covers |
> |---|---:|---:|---|
> | already-bought (v1, sunk) | 178 | $5 | Bahnhofstrasse strip only — 1 video well |
> | **targeted 150 m POI buffer** | ~800–1,000 | **$22–28** | every street the videos walk, dense |
> | tight 75 m POI buffer | ~400–600 | $11–17 | same routes, sparser, gaps likely |
> | blanket — every pano | 1,915 | $54 | whole bbox including streets never visited |
>
> **Iterate if needed.** After the targeted download, re-run
> `gps_recovery`. If `dino_weak` is still high (>20%), the remaining
> dino_weak frames' VLM-guess positions are exactly where the gap is —
> buffer those + a small top-up download (~100 panos, $3) fills it.
> "Data tells us where to spend" loop. Total expected spend ≈ **$25–35**
> vs $54 for blanket, with **fewer location-bias false positives** in
> `accepted` (we won't be over-biased toward over-represented streets).

### 2.5 GPS recovery

Each video frame needs a recovered **GPS** *and* **heading**. Neither
DINOv2 nor a VLM is reliable enough alone, so v2 combines two
independent estimates with a weighted score.

**Estimate 1 — DINOv2 visual match.** `dinov2-base` (CLS-token)
embeds the frame; cosine-match against the Street View index. Output:
the matched pano GPS `g_dino` and the best cosine similarity `s ∈ [0,1]`.
An absolute floor `min_sim` rejects weak matches (cosine similarity is
relative — without a floor an argmax always returns *something*).

> **DINOv2 match pilot — `src/dinov2_match.py`.** Before paying for a
> larger Street View crawl, we test how well DINOv2 actually matches
> v2 video frames against the **712 SV images already on disk** from
> v1 (178 panos × 4 headings). The script embeds both sets (CLS-token,
> cached to `data/cities/zurich/dinov2/*.npz`), computes top-K cosine,
> prints the top-1 distribution, and writes an HTML grid
> `viz/dinov2_match_test.html` partitioned by `--min-sim` (default 0.60)
> — **MATCHED** rows (top-1 ≥ threshold, sorted desc) and **NO MATCH**
> rows.
>
> **How the pilot was actually run** (reproducible):
>
> 1. **Copy the 712 SV images Drive → local SSD.** The originals live in
>    `G:\My Drive\cs231n\project\cs231n\cs231n\navlm_ss\data\cities\
>    streetview\zurich\images\`. An earlier run hung for 24 h reading
>    them over Drive, so we copy to the canonical v2 path
>    (`config.STREETVIEW_DIR / "images"`):
>    ```powershell
>    robocopy "G:\My Drive\cs231n\project\cs231n\cs231n\navlm_ss\data\cities\streetview\zurich\images" `
>             "C:\Users\z0502\Desktop\cs231n\navlm_v2\data\cities\streetview\zurich\images" `
>             /NFL /NDL /NJH /NJS /MT:8
>    # 712 jpgs, 60 MB
>    ```
>    The v1 `meta.jsonl` / `panos.jsonl` / `embeddings.npz` are **not**
>    copied — v2's extracted frames differ from v1, so the v1
>    embeddings are stale; meta is only needed once we wire GPS recovery.
> 2. **Create the local conda env.** The first attempts hung because the
>    venv lived on Google Drive (every Python import hit Drive). Fix:
>    ```powershell
>    cmd /c "call C:\Users\z0502\anaconda3\Scripts\activate.bat && `
>            conda create -n navlm_v2 -c conda-forge --override-channels `
>            python=3.11 pip -y"
>    # then install packages (see § How to run)
>    ```
> 3. **Run the pilot.** With `navlm_v2` activated (or via the env's
>    `python.exe` directly):
>    ```bash
>    python -m src.dinov2_match --every-n 40 --min-sim 0.60 -k 3
>    # ~655 v2 frames vs 712 SV refs, threshold cos >= 0.60
>    ```
>    Wall time ~1 min on an RTX 3060 (45 + 41 batches at ~3 batch/s).
>
> **Pilot results (first run, all 8 videos, every-40 sampling):**
>
> ```
> top-1 cosine: mean=0.586  median=0.617  min=0.089  max=0.844
>   > 0.85 very strong            0/655  ( 0%)
>   > 0.75 strong                79/655  (12%)
>   >= 0.60 MATCHED              352/655 (54%)   <- headline
>   > 0.50 weak                  484/655 (74%)
> ```
>
> **~54 % of frames matched** at cos ≥ 0.60 — the 712 v1 SV images cover
> about half the videos' routes. The remaining 46 % are the gap a fresh
> `--download` needs to fill. Max cos 0.844 (no near-identity) reflects
> viewpoint mismatch between phone-walk frames and SV's 4 fixed compass
> crops — exactly why the pipeline combines DINOv2 with VLM
> place-naming (§2.5 Estimate 2) instead of relying on cosine alone.

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

**Reconciliation — strict F1/F2/F3 (the current default).** A single
weighted-Q score was too forgiving — a high cosine + high VLM
confidence could drag a frame past `tau` even when the two GPSes
disagreed by hundreds of metres. Replaced with **three independent
filters; all three must pass to accept** (`src/reconcile.py:
reconcile_strict`, the legacy weighted version is kept as
`reconcile_weighted` for ablation):

```
F1.  cos_dino  >=  config.MIN_SIM                   (DINOv2 real visual match)
F2.  vlm_gps  is not None                            (VLM resolved to OSM POI)
F3.  exact-name match  OR
     distance(vlm_POI_geometry, dino_nearest_POI_geometry)
                                  <= config.NEIGHBORHOOD_RADIUS_M  (250 m)
```

F3 is **semantic agreement** (the *name* of the place VLM guessed
matches the OSM POI nearest to DINOv2's matched SV pano), with a
**neighborhood fallback** at 250 m (`distance_pois_m` uses point-to-line
distance via shapely, not centroid-to-centroid — so a long street like
Bahnhofstrasse is matched correctly anywhere along it). `src/spatial.py`
owns the geometric helpers + an STRtree on the OSM POI geometries.

> **Why semantic, not just spatial.** A naïve distance check
> (`|g_dino − g_vlm| ≤ 150 m`) was rescuing frames where the two
> signals were spatially close *by coincidence* — and a tighter
> threshold lost too many true positives because VLM's resolved POI is
> often a long street whose centroid is far from the photo. Comparing
> at the *name* level (with a 250 m buffer to absorb nearest-POI
> noise) is much more robust: for both estimates to converge on the
> same wrongly-named OSM place by accident, the visual and the
> linguistic failure modes have to correlate, which is statistically
> much rarer than positional coincidence.

**GPS on accept = `g_dino` always.** The SV pano is a real
photograph at a *known* coordinate (~5 m accuracy from Google).
The VLM's resolved POI is the *centroid* of an OSM feature —
acceptable for a small point POI, but kilometres off for long
features (Limmat, Zürichsee, Bahnhofstrasse). Trust the pano's
coords; the VLM's role here is purely to **confirm the place name
in F3**, not to contribute to the position. (`variance_m` =
`haversine(g_dino, g_vlm)` is still logged as a diagnostic, but
unused for the accepted GPS.)

**Heading — same-pano cosine-weighted mean.** Earlier we averaged
headings across the top-K SV crops, which mixed crops from *different
panos* (different locations). Now we commit to one location (top-1's
pano) and ask only "which direction at that pano?":

```
   heading  =  atan2( Σᵢ cosᵢ · sin θᵢ ,   Σᵢ cosᵢ · cos θᵢ )

where θᵢ ∈ {0°, 90°, 180°, 270°} are the 4 compass headings of the
SV crops at top-1's pano, and cosᵢ is the cosine similarity of the
query frame to each of those 4 crops (clamped to ≥0).
```

This is the cosine-weighted circular mean — a walker heading 45°
(where the pano's 0° and 90° crops both score high) gets `heading ≈
45°` instead of snapping to one of the four 90°-spaced discretized
directions.

**Heading confidence — `heading_gap`.** Per frame:

```
   gap  =  (best_at_pano − 2nd_at_pano) / best_at_pano
```

High gap (≥ 0.15) → DINOv2 *can* tell direction at this pano; low gap
(< 0.05) → 2+ directions look nearly identical, heading is genuinely
ambiguous (e.g., front-vs-back symmetric architecture). The 11 % of
accepted frames with low gap will get their heading **replaced** by
the segment bearing in the next (HMM) stage; the rest are trusted.

> **Why `heading_gap` rather than top-K circular spread?** Spread
> conflates two failure modes — *wrong pano* and *wrong direction at
> the right pano*. The same-pano gap isolates the second, which is the
> one that actually matters once F1/F2/F3 have locked in the location.

**Road-snapping (`src/road_snap.py`, ⏳).** The accepted per-frame GPS
sequence is smoothed onto the OSM walking graph with HMM map-matching
(Newson-Krumm Viterbi). Three things it does beyond per-frame
filtering:
- **Snap GPS** to walkable geometry (removes the small jitter from our
  ±30 m reconciled position).
- **Correct heading**: replace per-frame DINOv2 heading with the
  bearing of the chosen segment — fixes the 11 % `heading_gap < 0.05`
  ambiguous frames automatically.
- **Reject route outliers**: a frame whose snapped GPS is >50 m from
  the Viterbi path, or whose heading is >90° off the segment bearing,
  is the cross-frame disagreement check the per-frame stage cannot
  see. Eliminate or correct based on residuals.

Phase A output: **trusted frames, each with `(gps, heading, edge_id)`**.

**Per-frame stage measured on 872 scanned frames (Phase A frame-by-frame):**

```
accepted          258  (30%)   ← trustworthy frames
disagree          194  (22%)   ← genuine cross-bbox disagreements
dino_weak         395  (45%)   ← no good SV reference → fix by §2.4 targeted crawl
vlm_unresolved     25  ( 3%)   ← VLM named a real place missing from pois.json

heading_gap among 258 accepted:
  ≥ 0.15 confident   53%
  ≥ 0.05 some signal 89%
  <  0.05 ambiguous  11%   ← HMM will resolve via segment bearing

per-video accepted: 11–82 frames; eval hold-out (saturday_morning) 28
```

**Modules:** `src/gps_recovery.py` (main orchestrator) ·
`src/spatial.py` (POI geometry index + name match + neighborhood
distance) · `src/geo_check.py` (per-frame VLM → GPS, no live API call
when the frame is in `poi_scan.jsonl`) · `src/reconcile.py` (the
F1/F2/F3 logic + 50/50 blend + centroid-blend protection) ·
`src/viz_recovery_grid.py` (per-frame photo grid HTML — every frame
shows QUERY + the 4 compass crops at top-1's pano, with the chosen
direction outlined and the heading-calc math worked).

**Road-snapping.** The accepted per-frame GPS sequence is smoothed onto
the OSM walking graph with HMM map-matching (Newson-Krumm Viterbi),
removing jitter and forcing positions onto walkable geometry.

Phase A output: **trusted frames, each with (GPS, heading)**.

### 2.6 Routing — and turning a route into an instruction (Q5)

`src/routing.py` (`plan_route`, osmnx + networkx) computes the OSM
walking route from a frame's GPS to a destination POI:
`nx.shortest_path` by length, giving the route geometry and the
**first-segment absolute bearing `B`**.

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

**Run 5 samples first.** The annotation module `src/annotate.py` takes a
`--limit 5` flag; we inspect the 5 (thinking + answer + verifier verdict)
before the full run. It does the distance-banded destination sampling,
the closed-loop verifier, and the teacher call
`call_gemini(model="gemini-2.5-pro")`.

### 2.8 Image ↔ POI indexing & route map

Index: `poi_scan.jsonl` (§2.3) maps each video frame → the POIs visible
in it, resolved to the OSM table by name — so any frame is traceable
POI→frames and frame→POIs, which also yields the per-POI dataset
distribution.

Once frames carry GPS, plot the **8-video route map**: each video's
recovered GPS sequence as a coloured polyline on one Leaflet/folium map —
both a deliverable and a sanity check on GPS recovery.

### 2.9 Gemini API budget (Q6)

Per 1M tokens: Gemini 2.5 **Pro** $1.25 in / $10 out · **Flash**
$0.30 in / $2.50 out. All three VLM stages run on **Pro**, called
through **Vertex AI** (§2.3 *API backend*) — billed to project
`cs231n-navlm-2026`, so the Education credit applies. Vertex Pro pricing
matches the table above; **token usage and cost are logged per call to
`logs/gemini_api.jsonl`** (≈ $0.014/frame measured on the POI scan).

| Workload | Calls | Model | Est. cost |
|----------|------:|-------|----------:|
| POI scan — `--every-n 30`, 1024 px frames | ~870 | Pro / Vertex | ~$12 |
| VLM geo-check — candidate frames | ~5,000 | Pro / Vertex | ~$18 |
| Instruction annotation — 3 dest/frame | ~6,600 | Pro / Vertex | ~$76 |
| Street View Static crawl | ~2k–8k imgs | — | ~$15–55 |

Total ≈ **$120–160**, against the **$50 Education credit** — over
budget. Mitigate by trimming the Street View crawl to the video routes
and capping annotation samples. Flagged for the budget owner; live spend
is in `logs/gemini_api.jsonl` (Gemini) and `reference/track_spend.py`.

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

**Where it runs.** The zero-shot baselines (`B-given`, `B-implicit`,
`B-explicit`) run **locally** on the RTX 3060 — inference only, no
training. The three LoRA conditions are trained on Modal and evaluated
straight after each run. PASS_strict is compared across the 6 conditions
for **each** ablation.

*Prior round, for reference:* a compass-free explicit-CoT LoRA reached
52.9 % PASS / median heading error 20.8° (vs base 99°); the with-compass
ceiling was 100 %.

---

## 5. Visualizations

Standalone HTML in `viz/`:
1. **The 27 candidate POIs** with signature icons — `poi_candidates_map.html`
   (`python -m src.poi --map`). ✅ built.
2. **POI-scan map** — every matched POI from `poi_scan.jsonl` placed on
   a Leaflet map (folium), with the derived Street View crawl bbox(es)
   overlaid. ✅ built — `src/viz_scan.py` → `viz/poi_scan_map.html`.
   - dots: matched POIs · size ∝ log(sightings) · colour by tier
     (L1 red, L2 blue, L3 grey); click for name + reasoning from an
     example frame;
   - orange flags: *outlier* POIs whose OSM centroid sits outside
     `POI_BBOX` (rivers, lakes, long lake-front streets) — they pull the
     raw bbox out, so the recommended bbox excludes them;
   - rectangles: black = `POI_BBOX` (OSM extraction region); red solid
     = clean crawl bbox + 300 m (centroid-clipped, recommended);
     grey dashed = raw scan bbox + 300 m.
   - Run: `python -m src.viz_scan`.
3. **DINOv2 match grid** — v2 video frames matched against the 712 v1
   Street View images, rows sorted by top-1 cosine descending and
   split MATCHED / NO MATCH at the `--min-sim` threshold (default
   0.60). ✅ built — `src/dinov2_match.py` →
   `viz/dinov2_match_test.html`. The grid is the visual sanity check
   for "is the existing SV reference set sufficient for our routes?".
   Run: `python -m src.dinov2_match --every-n 40 --min-sim 0.60`.
4. **GPS-recovery map** — every reconciled frame on a Leaflet map,
   coloured by video for accepted; toggleable layers for `disagree`,
   `dino_weak`, `vlm_unresolved`. ✅ built — `src/viz_recovery.py` →
   `viz/gps_recovery_map.html`.
5. **GPS-recovery per-frame photo grid** — for each frame: QUERY photo
   + the **4 compass crops at top-1's pano** (the heading-decision
   evidence), with the chosen direction red-outlined; info panel with
   POI-to-POI distance, F3 outcome, `heading_gap`, and the **worked
   atan2 heading calculation** for that frame. ✅ built —
   `src/viz_recovery_grid.py` → `viz/gps_recovery_grid.html`.
   Sections: ACCEPTED / DISAGREE / VLM UNRESOLVED.
6. Bought Street View panos highlighted on the map.
7. Recovered video-frame GPS on the map, **overlaid with the POI region
   polygons (Q8)** — so each frame's area and POI assignment (§3) is
   visible at a glance.
8. The 8-video routes derived from the images (§2.8).
9. A Q&A viewer — photo + question + generated answer — to sanity-check
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

1. ✅ Scaffold — `config.py`, `src/`, `tests/` (95 pytest tests).
2. ✅ Phase A modules coded + unit-tested: `download_videos`,
   `extract_frames`, `pois`, `poi_scan`, `gemini_api`, `streetview`,
   `dinov2_match`, `gps_recovery`, `geo_check`, `spatial`, `reconcile`,
   `routing`, `road_snap` (stub); plus viz: `poi`, `viz`, `viz_scan`,
   `viz_recovery`, `viz_recovery_grid`; plus `annotate`, `train_modal`.
3. ✅ OSM POI table (`pois.json`, 1,289 POIs); frame extraction
   (26,034 kept frames).
4. ✅ POI scan — full run on Gemini 2.5 Pro via Vertex AI
   `--every-n 30` (872 frames, **227 distinct OSM POIs matched**,
   $10.68 of the Education credit). Crawl bbox derived
   (~4.0 × 4.4 km after centroid-clip; viz: `viz/poi_scan_map.html`).
5. ✅ DINOv2 match pilot — 712 v1 SV images, 55 % of frames matched at
   cos ≥ 0.60; remaining 45 % flag the SV coverage gap
   (`viz/dinov2_match_test.html`).
6. ✅ **GPS recovery (frame-by-frame)** — strict F1/F2/F3 +
   same-pano cosine-weighted heading + `heading_gap` (§2.5).
   **258 / 872 accepted (30 %)**, every video 11–82 frames, eval
   hold-out 28 frames. Viz: `viz/gps_recovery_map.html` +
   `viz/gps_recovery_grid.html`.
7. ▶ **Targeted Street View crawl (§2.4)** — buy panos within 150 m
   of the 227 matched POIs (~800–1,000 panos, ~$22–28) instead of
   the full 1,915 in the bbox. Iterate based on remaining
   `dino_weak` rate.
8. ⏳ **HMM road-snapping** (`src/road_snap.py`) — snap GPS to
   walking graph, **correct heading** from segment bearing, eliminate
   route-outlier frames. Produces `phaseA_trusted.jsonl`.
9. ⏳ 8-video route map + Q&A viewer (§5 items 8, 9).
10. ⏳ Phase B — routing + Gemini-2.5-Pro annotation (5-sample trial first).
11. ⏳ Phase C/D — `train_modal.py` LoRA runs + local zero-shot + eval,
    both ablations (§3 / §4).
