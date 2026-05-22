# NavLM end-to-end pipeline

Comprehensive reference for the full project flow:

```
raw .mp4 videos
     │
     ▼  Phase A — GPS RECOVERY  (per-video, then merged)
extract frames → DINOv2 embed → visual_match → refine →
ocr_paddle → landmark_match → merge_gps → HMM → vlm_poi_scan →
vlm_cross_verify → trusted_starts
     │
     ▼  Phase B — SYNTH DATA GENERATION
synth_unified (Gemma teacher) → step_12 closed-loop verify → strict filter
     │
     ▼  Phase C — TRAINING
train/eval split → LoRA SFT (Qwen2.5-VL-7B + r=16)
     │
     ▼  Phase D — EVALUATION
inference on hold-out → 6 gates (format/sentence/closed-loop/checkpoint/
  dest/anchor) → base vs LoRA comparison
```

End artifacts:
- `data/cities/zurich/frame_starts_trusted_all.jsonl`  (2,177 frames)
- `data/cities/zurich/synth_v3_full_strict.jsonl`       (4,689 train samples)
- `data/cities/zurich/synth_v3_train.jsonl` / `synth_v3_eval.jsonl`
- `results/lora_zurich_v3/`  (LoRA adapter weights)

---

## Repo layout

```
toolbox/                       Underlying tools (call directly)
  extract_frames.py            ffmpeg wrapper
  embed_images.py              DINOv2 embedding
  extract_frames.py            ffmpeg → per-second .jpg
  embed_images.py              DINOv2 ViT-L/14 embedding
  visual_match_gps.py          DINOv2 → Mapillary index nearest-neighbour
  refine_visual_match.py       gps/compass consensus → high/medium/low
  ocr_paddle.py                PaddleOCR per-frame text
  landmark_match.py            text + ZURICH_LANDMARKS table → GPS
  compute_frame_heading.py     Mapillary neighbour compass average
  way_planner.py               OSM walking route + relative actions
  scenery_pois.py              hand-curated street/river/lake POIs
  scan_video_pois_multi.py     Gemma per-frame POI visibility
  map_match.py                 HMM Newson-Krumm Viterbi
  extract_osm_pois.py          OSM landmark POI extraction
  zurich_landmarks_gps.py      ~50 hand-curated landmark GPS
  synth_unified.py             v3 prompt → Gemma teacher → jsonl
  synth/
    prompts.py                 SYSTEM_PROMPT, USER_TEMPLATE, helpers
    backends.py                Gemma/OpenAI/Anthropic dispatch
    verifier.py                light format checks
    sampling.py                tier-weighted destination sampler
  process_extra_videos.py      Phase A driver for raw → step 6 (new videos)
  fetch_*.py, expand_frames.py utility scripts
  draw_direction_arrow.py,     overlay helpers
  draw_poi_bbox.py
  _legacy/                     deprecated scripts (auto_annotate, build_sft, ...)

pipeline/                      Strict GPS-recovery driver (steps 7-11) + Phase B
  config.py                    8 videos, paths, TRUSTED_MATRIX
  step_07_merge_gps.py         visual_high + bbox sanity
  step_08_hmm.py               wraps map_match.py
  step_09_vlm_poi.py           splits combined VLM scan per video
  step_10_vlm_verify.py        PASS_LANDMARK/STREET/INCONCLUSIVE/FAIL
  step_11_trusted.py           matrix × HMM × heading_high
  step_12_closed_loop_verify.py  parse answer → check 6 gates
  run_all.py                   orchestrator (--from-step / --videos / --steps)
  build_v4_datasets.py         v3 → v4a (implicit) + v4b (explicit)
  build_v4c_rationale.py       v3 → v4c (Claude rationale)
  closed_loop_sanity[_v2].py   math validation
  report.py                    trusted_starts breakdown stats
  PIPELINE.md                  this file

scripts/                       Phase C train + Phase D eval + portals
  train_lora_cot.py            Qwen2.5-VL LoRA SFT
  eval_lora.py                 base vs LoRA on hold-out, 6 gates
  plot_eval_comparison.py      bar charts of all conditions
  visualize_paths.py           folium 8-video map
  synth_viewer.py              Flask :9000 portal
                               (/, /map, /experiment_summary, /experiment)
  _legacy/                     deprecated scripts (eval_synth, train_lora, ...)

data/                          (gitignored)
  raw/                         original .mp4 files
  cities/zurich/
    frames/                    extracted .jpg per video
    frame_gps*.jsonl           per-video output of phase A
    pipeline/<video>/          step_07/08/09/10/11 per-video
    frame_starts_trusted_all.jsonl  ⭐ Phase A endpoint (2,177 frames)
    synth_v3_full.jsonl        v3 raw (6,510)
    synth_v3_full_verified.jsonl   passed 4 step-12 gates (6,249)
    synth_v3_full_strict.jsonl ⭐ training pool, δ<30° (4,689)
    synth_v3_train.jsonl       4,434 (saturday_morning excluded)
    synth_v3_eval.jsonl        255 hold-out
    synth_v4a_train/eval.jsonl  derived (no heading, implicit)
    synth_v4b_train/eval.jsonl  derived (no heading, explicit + INFERRED_HEADING)
    synth_v4c_train/eval.jsonl  Claude rationale-grounded
  mapillary/zurich_full/       embedding index + meta + images
  mapillary/zurich_altstadt/   (planned dense10k from laptop)

results/                       Phase C / Phase D outputs
  EXPERIMENT_REPORT.md         ⭐ 6-condition writeup (H1-H7)
  lora_zurich_v3/              EXP-C1 weights
  lora_zurich_v4a/             EXP-C2 weights
  lora_zurich_v4b/             EXP-C3 weights
  lora_zurich_v4c/             EXP-C4 weights (training)
  eval_v3_<tag>.json           summary per condition
  eval_v3_<tag>.jsonl          per-row details
  plot_*.png                   4 comparison bar charts
  _legacy/                     historical demo files (mp3, mp4, jpg)

README.md → pipeline/PIPELINE.md     symlink
```

### Repo cleanup history

```
2026-04-27  Repo restructure
  - 12 deprecated scripts → toolbox/_legacy/
    (auto_annotate, build_eval_map_context, build_ood_from_photos, build_sft,
     build_trusted_starts, merge_gps_sources, ocr_augment,
     scan_mapillary_landmarks, scan_video_pois, scene_inventory,
     temporal_smooth, train_city_lora)
  - 13 deprecated scripts → scripts/_legacy/
    (baseline_demo, compare_base_vs_lora, draw_direction, eval_hallucination,
     eval_image_text, eval_synth, eval_video_speech, extend_video,
     gpu_queue, prepare_cot_sft, prepare_sft_data, train_lora, voice_demo)
  - results/{cogvideox_test.mp4, comparison.*, e2e_*} → results/_legacy/
  - scripts/eval_lora_v3.py → scripts/eval_lora.py  (refs updated)
  - README.md → pipeline/PIPELINE.md  (symlink)
```

---

# Phase A — GPS RECOVERY (raw video → trusted_starts)

## Step 1 — extract frames

- **What**: ffmpeg-wraps each .mp4, extract one frame per second as .jpg.
- **Tool**: `toolbox/extract_frames.py`
- **Input**:  `data/raw/<video>.mp4`
- **Output**: `data/cities/zurich/frames/<video>/frame_NNNNN.jpg`
- **Command**:
  ```bash
  python toolbox/extract_frames.py \
      --video data/raw/Zurich_Old_Town.mp4 \
      --out   data/cities/zurich/frames/zurich
  ```
- **Notes**: Already done for the 8 videos. Re-run only if a video is added.

## Step 2 — DINOv2 embeddings

- **What**: Load each frame, compute 1024-d DINOv2 (ViT-L/14) embedding.
- **Tool**: `toolbox/embed_images.py`
- **Input**:  `frames/<video>/*.jpg`
- **Output**: `frames/<video>_embeddings.npz`  (one big numpy)
- **Command**:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python toolbox/embed_images.py \
      --frames frames/zurich \
      --out    frames/zurich_embeddings.npz
  ```
- **Time**: ~10 min/video on L20X. Already done for all 8.

## Step 3 — visual match against Mapillary index

- **What**: For each frame, find top-5 most similar Mapillary images (cosine).
  Each Mapillary frame has known GPS, so we get 5 candidate GPS points.
- **Tool**: `toolbox/visual_match_gps.py`
- **Input**:  frame embeddings + Mapillary embedding index (`data/mapillary/zurich_full/`)
- **Output**: `<video>_frame_gps.jsonl` with `top_matches: [{id, lat, lon, sim}, …]`
- **Command**: orchestrated by `process_extra_videos.py`; manual:
  ```bash
  python toolbox/visual_match_gps.py \
      --query-embeds frames/zurich_embeddings.npz \
      --mly-embeds   data/mapillary/zurich_full/embeddings.npz \
      --mly-meta     data/mapillary/zurich_full/meta.jsonl \
      --out          frame_gps.jsonl
  ```

## Step 4 — refine visual match

- **What**: Tag each frame's top-5 with confidence based on dispersion.
- **Tool**: `toolbox/refine_visual_match.py`
- **Output**: `<video>_frame_gps_refined.jsonl` with new fields:
  - `gps_dispersion_m`     — max pairwise distance among top-5
  - `compass_spread_deg`   — top-5 compass circular std
  - `n_inliers`            — count surviving outlier filter
  - `confidence_v2`        — `high` / `medium` / `low`
- **Thresholds**:

  | tier   | gps_disp ≤ | compass_spread ≤ | n_inliers ≥ |
  |--------|------------|------------------|-------------|
  | high   | 50 m       | 30°              | 3           |
  | medium | 120 m      | 60°              | 3           |
  | low    | else       | else             | <3          |

- **Command**:
  ```bash
  python toolbox/refine_visual_match.py \
      --frame-gps frame_gps.jsonl \
      --mly-meta  data/mapillary/zurich_full/meta.jsonl \
      --out       frame_gps_refined.jsonl
  ```

## Step 5 — PaddleOCR

- **What**: Run PaddleOCR over each frame; collect raw text strings.
- **Tool**: `toolbox/ocr_paddle.py`  (replaces older Gemma OCR which was 25× slower)
- **Output**: `<video>_frame_ocr.jsonl`  with `texts_seen: [str, ...]`
- **Command**:
  ```bash
  python toolbox/ocr_paddle.py \
      --frames frames/zurich \
      --out    frame_ocr.jsonl
  ```
- **Time**: ~64 min for 27k frames on CPU.

## Step 6 — landmark match (OCR text → GPS)

- **What**: Compare OCR text against the hand-curated `ZURICH_LANDMARKS`
  table; if a landmark name appears in the text, attach that landmark's
  GPS as the frame's GPS.
- **Tool**: `toolbox/landmark_match.py`
- **Output**: `<video>_frame_gps_ocr.jsonl` with `matched: [name]`, `gps: [lat, lon]`
- **Note**: Used as **evidence only** in v3 (not a primary GPS source) —
  see step 7 — because of the airport-shuttle-ad bug.

---

# Pipeline driver: `pipeline/`

Steps 7–11 are encapsulated in the `pipeline/` package. Run via:

```bash
# Run everything from step 7 across all 8 videos
python -m pipeline.run_all --from-step 7

# Or run a specific step on a specific video
python -m pipeline.step_07_merge_gps --video bahnhofstrasse
```

## Step 7 — merge_gps (strict)

- **What**: Pick the best GPS per frame from refined visual match.
- **Module**: `pipeline/step_07_merge_gps.py`
- **Policy** (current strict version):
  1. Keep ONLY `confidence_v2 = high`.  (visual_medium / visual_low / OCR-alone all dropped.)
  2. OCR matched landmarks recorded in `evidence.ocr_matched` but do NOT supply GPS.
  3. Bbox sanity: drop GPS more than 1.5 km from old-town centre (47.374, 8.541).
- **Input**:  `<video>_frame_gps_refined.jsonl`, `<video>_frame_gps_ocr.jsonl`
- **Output**: `pipeline/<video>/step_07_merged.jsonl`
- **Result**: 27,075 → ~2,553 frames across 8 videos.

## Step 8 — HMM map matching

- **What**: Snap each frame's GPS to the OSM walking graph via Newson-Krumm Viterbi.
- **Module**: `pipeline/step_08_hmm.py` wraps `toolbox/map_match.py`.
- **Algorithm**:
  - emission: Gaussian on perpendicular distance to edge (σ = 20 m)
  - transition: `exp(-|gc_dist - osm_dist| / β)` (β = 10 m)
  - per-frame top-K candidate edges within 80 m
- **Output**: `pipeline/<video>/step_08_hmm.jsonl` per frame:
  `{frame_id, edge: [u, v, k], snap_lat, snap_lon, perp_m, confidence, matched}`
- **Result**: 100% match rate on 2,553 high-confidence frames; mean perp ~10 m.

## Step 9 — VLM POI scan (split)

- **What**: Splits the combined Gemma POI scan output into per-video files.
- **Module**: `pipeline/step_09_vlm_poi.py`
- **Source**: `data/cities/zurich/_video_poi_multi.jsonl` — produced once
  by `toolbox/scan_video_pois_multi.py` which scanned every-20th frame
  for visible POIs from a 26-candidate list.
- **Output**: `pipeline/<video>/step_09_vlm_poi.jsonl`  with `visible_pois: [...]`

## Step 10 — VLM cross-verify

- **What**: Determine if HMM-snap GPS is consistent with what Gemma saw.
- **Module**: `pipeline/step_10_vlm_verify.py`
- **Logic** (no VLM call — pure set intersection):
  ```
  near    = POIs within 50 m of HMM-snap
  visible = POIs within 200 m
  overlap = vlm_pois ∩ visible
  if not vlm_pois:    PASS_STREET if not near, else INCONCLUSIVE
  if overlap:         PASS_LANDMARK
  if min(vlm_dist) > 500m: FAIL
  else: INCONCLUSIVE
  ```

## Step 11 — trusted_starts filter

- **What**: Apply (gps_source × verdict) matrix + heading filter.
- **Module**: `pipeline/step_11_trusted.py`
- **Three gates per frame** (must all pass):
  1. `TRUSTED_MATRIX[gps_source][verdict] == True`
     (currently: only `visual_high` row exists; FAIL drops everything)
  2. HMM `matched == True`
  3. `heading_confidence == "high"`  (compass spread ≤ 30°)
- **Output**: `pipeline/<video>/step_11_trusted.jsonl` per video, then
  combined into `data/cities/zurich/frame_starts_trusted_all.jsonl`.
- **Result**: **2,177 trusted frames** across 8 videos (down from 27,075 raw).

---

# Phase B — SYNTH DATA GENERATION

## synth_unified (v3 prompt)

- **Tool**: `toolbox/synth_unified.py`
- **Input**:
  - `data/cities/zurich/frame_starts_trusted_all.jsonl`  (2,177 frames)
  - `data/cities/zurich/landmarks_zurich_osm.json` + `scenery_pois.py`
  - `data/cities/zurich/osm_walking.pkl` (walking graph)
- **Per frame**:
  1. Sample N destinations (tier-weighted: 70 % iconic / 25 % mid / 5 % small),
     restricted to `[150 m, 1500 m]`.
  2. Run `way_planner.plan(start, dest, user_heading=...)` → route.
  3. Format USER_TEMPLATE (heading + nearby POIs + first_seg_bearing +
     route info + natural-language question).
  4. Call Gemma teacher with image + system + user.
  5. Parse `<thinking>` and `<answer>`.
  6. Light verifier (just for early skip): both blocks present, no forbidden
     terms in answer, sentence count 2-4. Heavy checks come at step 12.
- **Output**: `data/cities/zurich/synth_v3_full.jsonl`
- **Command**:
  ```bash
  python toolbox/synth_unified.py \
      --starts data/cities/zurich/frame_starts_trusted_all.jsonl \
      --out    data/cities/zurich/synth_v3_full.jsonl \
      --n-dest-per-frame 3 \
      --skip-visual-verify \
      --backend gemma \
      --seed 42
  ```
- **Time**: ~3 s / sample on Gemma. 2,177 × 3 ≈ 6 hours.
- **Result**: 6,510 emitted (out of 6,531 = 99.7 %).

## V3 prompt design

- **System**: model is given heading explicitly and the first-segment
  absolute bearing. It must compute the relative action verb (the math
  is shown step-by-step in `STEP 3` of the CoT), then ground the answer
  in a visible object from the photo.
- **Output structure**:
  ```
  <thinking>
    STEP 1 (understand the question): ...
    STEP 2 (resolve coordinates): ...
    STEP 3 (compute the relative action): explicit arithmetic
    STEP 4 (look at the image): list visible objects with LEFT/CENTER/RIGHT
    STEP 5 (pick an anchor): ...
    STEP 6 (plan the answer): ...
  </thinking>
  <answer>
    2-4 short TTS-friendly sentences. Uses the action verb from STEP 3
    in the first sentence. References the anchor from STEP 5.
    For long routes, ends with "send me another photo at <CHECKPOINT>"
    where CHECKPOINT is a route-derived street name (mode B) or a
    permanent landmark (mode A fallback).
  </answer>
  ```

## Step 12 — closed-loop verifier

- **Module**: `pipeline/step_12_closed_loop_verify.py`
- **Six gates**:

  | gate | check | cost |
  |------|-------|------|
  | 1_format          | no compass / metres / GPS / em-dash / bullets | regex |
  | 2_sentence_count  | 2-4 sentences, ≤22 words each                  | regex |
  | 3_closed_loop     | heading_gt + ACTION_DELTA[parsed_verb] ≈ first_seg_bearing (δ ≤ 55°) | math |
  | 4_checkpoint      | mentioned street is in route, OR uses permanent-landmark word | regex |
  | 5_dest_correct    | destination POI name appears in answer | regex |
  | 6_anchor_grounded | independent VLM (Gemma) confirms answer's visual references | ~3 s/sample |

- **Command**:
  ```bash
  python -m pipeline.step_12_closed_loop_verify \
      --in  data/cities/zurich/synth_v3_full.jsonl \
      --out data/cities/zurich/synth_v3_full_verified.jsonl
  ```
- **Result on synth_v3_full**:
  - all-4-gates (without 5/6): 6,249 / 6,510 = 96.0 %
  - δ < 30 ° subset:           4,876 (74.9 %)

## Strict filter

- **What**: Keep only samples that pass all gates AND have δ < 30°.
- **Output**: `data/cities/zurich/synth_v3_full_strict.jsonl`  (4,689 samples)

---

# Phase C — TRAINING

## train/eval split

- **Strategy**: hold out one entire video (`saturday_morning`) so the model
  never sees its frames during training.
- **Result**:
  - `synth_v3_train.jsonl`  4,434 samples
  - `synth_v3_eval.jsonl`     255 samples

## LoRA SFT

- **Script**: `scripts/train_lora_cot.py`
- **Base**: `Qwen2.5-VL-7B-Instruct`
- **Adapter**: LoRA r=16, alpha=32 → 47.6 M trainable params (0.57 %)
- **Image cap**: `max_pixels = 448²` (cuts memory ~3× vs full-res)
- **Optimizer**: bf16, sdpa attention, batch=1 × grad_accum=8 = effective 8
- **Schedule**: 2 epochs, lr 2e-4, ~1110 steps total
- **Command**:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/train_lora_cot.py \
      --train  data/cities/zurich/synth_v3_train.jsonl \
      --val    data/cities/zurich/synth_v3_eval.jsonl \
      --output results/lora_zurich_v3 \
      --epochs 2 --batch_size 1 --grad_accum 8 \
      --lr 2e-4 --lora_r 16 --lora_alpha 32
  ```
- **Time**: ~6 h on a single L20X.
- **Output**: `results/lora_zurich_v3/` — checkpoint dirs + train.log

---

# Phase D — EVALUATION

## eval_lora

- **Script**: `scripts/eval_lora.py`  (was `eval_lora_v3.py` pre-2026-04-27)
- **What**: For each sample in the hold-out:
  1. Run inference with the model (base or base+LoRA).
  2. Apply step 12 verifier's six gates to the model's output.
  3. Aggregate per-gate pass rates + closed-loop δ distribution.
- **PASS_strict** = all CORE gates pass AND `δ < 30°`. CORE gates =
  {format, sentence_count, closed_loop, checkpoint, anchor_grounded}.
  `dest_correct` (gate 5) is reported but does NOT block PASS, since
  it's a naming-style check rather than a correctness check.
- **Commands**:
  ```bash
  # base only
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_lora.py \
      --eval data/cities/zurich/synth_v3_eval.jsonl \
      --tag  base_v3

  # with LoRA (one tag per condition)
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_lora.py \
      --eval data/cities/zurich/synth_v3_eval.jsonl \
      --lora results/lora_zurich_v3 \
      --tag  lora_v3

  # side-by-side comparison
  python scripts/eval_lora.py --compare \
      results/eval_v3_base_v3.json results/eval_v3_lora_v3.json
  ```
- **Time per run**: ~30 min inference + ~13 min Gemma anchor check =
  ~45 min total. With `--skip-hallucination` it's ~30 min.

## plot_eval_comparison

- **Script**: `scripts/plot_eval_comparison.py`
- **Output**: 4 PNGs in `results/`
  - `plot_pass_strict.png` — overall pass rate per condition
  - `plot_closed_loop.png` — gate 3 only (geometric correctness)
  - `plot_gate_pass_rates.png` — 6 gates × N conditions grouped bars
  - `plot_delta_distribution.png` — δ band stacking
- **Run**:
  ```bash
  python scripts/plot_eval_comparison.py
  ```

## Visualization viewer

- **Script**: `scripts/synth_viewer.py`
- **What**: Flask server on `:9000` with four sections:
  - `/` — browse synth_unified samples (image + thinking + answer + TTS)
  - `/map` — folium 8-video paths on Zurich
  - `/experiment_summary` — 6-condition table + 4 plots inline
  - `/experiment` — 255 hold-out frames list
  - `/experiment/<frame_id>` — same image, side-by-side 6 model outputs +
    per-gate badges
- **Run**:
  ```bash
  python scripts/synth_viewer.py \
      --jsonl data/cities/zurich/synth_v3_full_strict.jsonl
  ```
  Then open `http://localhost:9000` (SSH-tunnel friendly:
  `ssh -L 9000:localhost:9000 root@<host>`).

---

# Quick start commands (re-run from scratch)

## Phase A — GPS recovery (assumes raw frames already extracted)

```bash
python -m pipeline.run_all --from-step 7
```

## Phase B — Synth data generation

```bash
# 1. Generate v3 synth data (~6 hours, ~$50 if Gemma teacher local)
python toolbox/synth_unified.py \
    --starts data/cities/zurich/frame_starts_trusted_all.jsonl \
    --out    data/cities/zurich/synth_v3_full.jsonl \
    --n-dest-per-frame 3 --skip-visual-verify --backend gemma

# 2. Verify + filter to strict δ<30° training pool
python -m pipeline.step_12_closed_loop_verify \
    --in  data/cities/zurich/synth_v3_full.jsonl \
    --out data/cities/zurich/synth_v3_full_verified.jsonl

# 3. Derive v4a (implicit) and v4b (explicit, INFERRED_HEADING) from v3
python pipeline/build_v4_datasets.py

# 4. (optional) Generate v4c with Claude rationale (~$50, ~1.5h)
export ANTHROPIC_API_KEY='sk-ant-api03-...'
python pipeline/build_v4c_rationale.py --workers 8
```

## Phase C — Train (one LoRA per condition)

```bash
# C1: with heading
CUDA_VISIBLE_DEVICES=1 python scripts/train_lora_cot.py \
    --train  data/cities/zurich/synth_v3_train.jsonl \
    --val    data/cities/zurich/synth_v3_eval.jsonl \
    --output results/lora_zurich_v3 \
    --epochs 2 --batch_size 1 --grad_accum 8 \
    --lr 2e-4 --lora_r 16 --lora_alpha 32

# C2: no heading, implicit CoT
CUDA_VISIBLE_DEVICES=1 python scripts/train_lora_cot.py \
    --train  data/cities/zurich/synth_v4a_train.jsonl \
    --val    data/cities/zurich/synth_v4a_eval.jsonl \
    --output results/lora_zurich_v4a --epochs 2 --batch_size 1 --grad_accum 8

# C3: no heading, explicit CoT (with INFERRED_HEADING step)
CUDA_VISIBLE_DEVICES=2 python scripts/train_lora_cot.py \
    --train  data/cities/zurich/synth_v4b_train.jsonl \
    --val    data/cities/zurich/synth_v4b_eval.jsonl \
    --output results/lora_zurich_v4b --epochs 2 --batch_size 1 --grad_accum 8

# C4: no heading, Claude rationale
CUDA_VISIBLE_DEVICES=2 python scripts/train_lora_cot.py \
    --train  data/cities/zurich/synth_v4c_train.jsonl \
    --val    data/cities/zurich/synth_v4c_eval.jsonl \
    --output results/lora_zurich_v4c --epochs 2 --batch_size 1 --grad_accum 8
```

## Phase D — Evaluate (6 conditions on the same hold-out)

```bash
# Base evals (no finetune, just inference)
CUDA_VISIBLE_DEVICES=3 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v3_eval.jsonl  --tag base_v3  --skip-hallucination
CUDA_VISIBLE_DEVICES=3 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v4a_eval.jsonl --tag base_v4a --skip-hallucination
CUDA_VISIBLE_DEVICES=3 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v4b_eval.jsonl --tag base_v4b --skip-hallucination

# LoRA evals (one per condition, eval set must match training prompt format)
CUDA_VISIBLE_DEVICES=3 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v3_eval.jsonl  --lora results/lora_zurich_v3  --tag lora_v3  --skip-hallucination
CUDA_VISIBLE_DEVICES=3 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v4a_eval.jsonl --lora results/lora_zurich_v4a --tag lora_v4a --skip-hallucination
CUDA_VISIBLE_DEVICES=3 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v4b_eval.jsonl --lora results/lora_zurich_v4b --tag lora_v4b --skip-hallucination
CUDA_VISIBLE_DEVICES=3 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v4c_eval.jsonl --lora results/lora_zurich_v4c --tag lora_v4c --skip-hallucination

# Render bar charts + open viewer
python scripts/plot_eval_comparison.py
python scripts/synth_viewer.py --jsonl data/cities/zurich/synth_v3_full_strict.jsonl
```

---

# Current numbers (as of run 2026-04-27)

| stage | size |
|-------|------|
| 8 raw videos | ~3 GB |
| Extracted frames | 27,075 |
| Visual matches (high confidence) | 2,553 |
| trusted_starts after HMM + heading filter | 2,177 |
| Synth (Gemma teacher, v3 prompt) | 6,510 |
| Step 12 verified (4 gates) | 6,249 (96.0 %) |
| Strict δ < 30° subset | 4,689 |
| Training set | 4,434 |
| Hold-out eval set | 255 (saturday_morning) |
