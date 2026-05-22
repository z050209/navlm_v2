# AGENTS.md — orient yourself in 5 minutes

Read this first. You are a Claude agent picking up the NavLM project
with no prior context. This file tells you:
1. what the project is doing,
2. where the important files live,
3. how to do the four common tasks,
4. what NOT to do.

If anything in another `.md` file conflicts with this one, **trust this
one** — other docs are snapshots from older states.

---

## 1. 30-second project summary

NavLM trains a small VLM (Qwen2.5-VL-7B) to give walking directions to
a lost traveler. The model takes:
- a photograph from the user's phone,
- the user's GPS,
- (optionally) the user's compass heading,
- a list of nearby POIs with absolute coordinates,
- an OSM-planned route described in absolute compass bearings,
- a natural-language question ("how do I get to Paradeplatz?").

It outputs a 2–4 sentence TTS-friendly answer that uses **relative**
verbs (turn left, continue, turn around, turn right) anchored to
**objects visible in the photo** ("turn left at the tram tracks").

The thesis is that the VLM can be fine-tuned to **infer the missing
heading from the photograph**, by triangulating between visible
landmarks and the POI map. This way, a deployed system doesn't need a
reliable phone compass — which fails in dense old-town blocks.

```
   user phone                          OSM backend
  ┌───────────┐                      ┌───────────────┐
  │ camera    │                      │ way_planner   │
  │ GPS       │                      │ POI lookup    │
  │ question  │                      └───────┬───────┘
  └─────┬─────┘                              │
        │                                    │
        ▼                                    ▼
        ┌──────────────────────────────────────┐
        │         Qwen2.5-VL-7B + LoRA          │
        │  (this project's contribution)        │
        └──────────────────┬───────────────────┘
                           ▼
              "Turn left at the tram tracks.
               Walk along Bahnhofstrasse, that's the
               main shopping street, until you reach
               Paradeplatz, a small public square."
                           │
                           ▼
                       (TTS plays it)
```

The **innovation** is not the architecture — it's the **self-supervised
data pipeline** that makes 4,689 high-quality training samples from 8
unlabelled YouTube walking-tour videos with no manual GPS annotation.

---

## 2. The four common tasks

### Task A: "I want to see the experimental results"

```
results/EXPERIMENT_REPORT.md             ← read this
results/plot_*.png                       ← 4 bar charts
results/eval_v3_*.json/.jsonl            ← raw per-condition data (7 conditions)
```

**Headline numbers** (5-gate PASS_strict, hold-out = saturday_morning,
255 samples):

| ID  | model           | prompt          | heading in user msg? | PASS_strict | median δ |
|-----|-----------------|-----------------|----------------------|-------------|----------|
| A1  | base Qwen-7B    | v3              | yes                  | 38.0%       | 80.7°    |
| A2  | base Qwen-7B    | v4a (implicit)  | no                   | 23.1%       | 89.8°    |
| A3  | base Qwen-7B    | v4b (explicit)  | no                   | 9.0%        | 98.7°    |
| C1  | LoRA            | v3              | yes                  | 100.0%      | 8.9°     |
| C2  | LoRA            | v4a (implicit)  | no                   | 46.3%       | 66.1°    |
| C3  | LoRA            | v4b (explicit)  | no                   | **52.9%**   | **20.8°** |
| C4  | LoRA            | v4c (Claude CoT)| no                   | 41.6%       | 66.2°    |

The point: C3 is the most interesting — fine-tuned with explicit
chain-of-thought, no compass given, achieves median 21° heading error
(vs base 99°).

### Task B: "I want to evaluate the shipped LoRAs"

```bash
# 30 minutes per LoRA on a single 80+ GB GPU
CUDA_VISIBLE_DEVICES=0 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v3_eval.jsonl \
    --lora results/lora_zurich_v3 \
    --tag  lora_v3_repro \
    --skip-hallucination
```

Read `REPRODUCE.md` section 1 for full L1 recipe.

### Task C: "I want to re-train a LoRA"

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_cot.py \
    --train  data/cities/zurich/synth_v3_train.jsonl \
    --val    data/cities/zurich/synth_v3_eval.jsonl \
    --output results/lora_zurich_v3_repro \
    --epochs 2 --batch_size 1 --grad_accum 8 \
    --lr 2e-4 --lora_r 16 --lora_alpha 32
```

3 hours on one L20X / A100 80 GB. Read `REPRODUCE.md` section 2.

### Task D: "I want to understand the data pipeline"

```
pipeline/PIPELINE.md     ← READ THIS, it's the canonical pipeline doc
                            (12 steps, raw video → trusted_starts → synth → train → eval)
```

---

## 3. File map — what's important, what to ignore

```
navlm_ss/
├── README.md                        Top-level overview, quick start
├── README_navlm.md                  Old top-level README (less useful)
├── REPRODUCE.md                ★    Step-by-step reproduction commands
├── AGENTS.md                   ★    THIS FILE
├── requirements.txt
│
├── pipeline/                        ★ Phase A→D driver code
│   ├── PIPELINE.md             ★    Detailed 12-step pipeline doc
│   ├── config.py                    Video list, paths, thresholds
│   ├── step_07_merge_gps.py    ─┐
│   ├── step_08_hmm.py            │  Phase A (GPS recovery, steps 7-11)
│   ├── step_09_vlm_poi.py        │
│   ├── step_10_vlm_verify.py     │
│   ├── step_11_trusted.py      ─┘
│   ├── step_12_closed_loop_verify.py   Verifier for synth and eval
│   ├── run_all.py                   Phase A driver
│   ├── build_v4_datasets.py         Derive v4a + v4b from v3
│   ├── build_v4c_rationale.py       Derive v4c via Claude API
│   ├── PROMPT_DRAFT.md              Historical scratch — IGNORE
│   ├── closed_loop_sanity.py        ── ditto, smoke-test scripts
│   ├── closed_loop_sanity_v2.py     ──
│   └── report.py                    Stats helper
│
├── toolbox/                         ★ Phase A tools (called by pipeline/)
│   ├── extract_frames.py            ffmpeg → 1 fps jpg
│   ├── embed_images.py              DINOv2 ViT-L/14 → embeddings.npz
│   ├── visual_match_gps.py          DINOv2 NN against Mapillary index
│   ├── refine_visual_match.py       gps + compass consensus → high/medium/low
│   ├── ocr_paddle.py                PaddleOCR per-frame text
│   ├── landmark_match.py            text → ZURICH_LANDMARKS table → GPS
│   ├── compute_frame_heading.py     Mapillary neighbour compass average
│   ├── way_planner.py               OSM walking-route planner (osmnx + nx)
│   ├── scenery_pois.py              hand-curated streets / rivers / lakes
│   ├── scan_video_pois_multi.py     Gemma per-frame POI visibility
│   ├── map_match.py                 HMM Newson-Krumm Viterbi
│   ├── synth_unified.py             ★ generates training data (calls Gemma teacher)
│   ├── synth/
│   │   ├── prompts.py          ★    SYSTEM_PROMPT + USER_TEMPLATE
│   │   ├── backends.py              gemma / openai / anthropic dispatch
│   │   ├── verifier.py              light format checks
│   │   └── sampling.py              POI tier weighting
│   ├── process_extra_videos.py      Phase A driver for new videos
│   ├── extract_osm_pois.py          OSM POI table builder
│   ├── zurich_landmarks_gps.py      ~50 hand-curated landmarks
│   ├── draw_*.py                    overlay helpers (not core)
│   └── (do NOT trust file-header doctstrings that mention "v2 design"
│        — most code is on v3; check synth/prompts.py for current prompt)
│
├── scripts/                         ★ Train + eval + viz (Phase C, D)
│   ├── train_lora_cot.py            Qwen-VL LoRA SFT
│   ├── eval_lora.py            ★    base vs LoRA, 6 gates
│   ├── plot_eval_comparison.py      bar charts
│   ├── visualize_paths.py           folium map of 8-video paths
│   ├── synth_viewer.py              Flask :9000 portal
│   └── serve_teacher.sh        ★    one-command vLLM launcher
│
├── docs/
│   └── TEACHER_DEPLOY.md       ★    Gemma + Qwen3-VL vLLM deployment
│
├── draft/                           NeurIPS 2026 paper draft
│   ├── main.tex + introduction.tex + related_work.tex + gps_data_pipeline.tex
│   ├── paper.md                ★    full markdown sketch (use this for context)
│   └── neurips_2026.{sty,tex}
│
├── data/
│   ├── cities/zurich/
│   │   ├── frames/<8 videos>/                ★ 27,075 jpg (~13 GB)
│   │   ├── frame_starts_trusted_all.jsonl    ★ Phase A endpoint (2,177 frames)
│   │   ├── synth_v3_*.jsonl                  ★ training datasets (see §4 below)
│   │   ├── synth_v4a_*.jsonl                       (no heading, implicit CoT)
│   │   ├── synth_v4b_*.jsonl                       (no heading, explicit CoT)
│   │   ├── synth_v4c_*.jsonl                       (no heading, Claude rationale)
│   │   ├── landmarks_zurich_osm.json         ★ POI table
│   │   ├── osm_walking.pkl                   ★ OSM walking graph
│   │   └── _video_poi_multi.jsonl                  Gemma POI scan
│   ├── cities/mapillary/zurich/              ★ 5k Mapillary index (655 MB)
│   └── mapillary/zurich_full/                meta.jsonl only (images dropped)
│
└── results/                         ★ trained weights + eval artefacts
    ├── EXPERIMENT_REPORT.md    ★    H1-H7 hypotheses + numbers
    ├── lora_zurich_v3/         ★    C1: 100% PASS (heading given)
    ├── lora_zurich_v4a/             C2: 46.3% PASS (implicit, no heading)
    ├── lora_zurich_v4b/        ★    C3: 52.9% PASS (explicit, no heading) — best compass-free
    ├── lora_zurich_v4c/             C4: 41.6% PASS (Claude rationale)
    ├── eval_v3_*.json + .jsonl      7 condition summaries
    └── plot_*.png                   4 comparison bar charts
```

★ = file you'll likely need.

---

## 4. v3 / v4a / v4b / v4c — what each dataset corresponds to

These are NOT software version numbers. They are **four prompt variants
used in the experiment**, each producing its own training set and LoRA:

| dataset                       | corresponds to | what it does                                            |
|-------------------------------|----------------|---------------------------------------------------------|
| `synth_v3_train.jsonl`        | EXP-C1         | user message **includes** heading; baseline             |
| `synth_v4a_train.jsonl`       | EXP-C2         | strip heading from user msg, **implicit** target (just copy v3 answer) |
| `synth_v4b_train.jsonl`       | EXP-C3         | strip heading from user msg, **explicit** CoT target with `INFERRED_HEADING:` step |
| `synth_v4c_train.jsonl`       | EXP-C4         | strip heading, explicit CoT, but the rationale is regenerated by **Claude Sonnet 4.6** for richer reasoning |

The matching `synth_v?_eval.jsonl` is the same 255-frame `saturday_morning`
hold-out — only the prompt format differs.

The `synth_v3_full_strict.jsonl` (4,689) is the **upstream pool** from
which v4a / v4b / v4c are mechanically derived (regex transformation).
Do NOT re-derive; the four files are already shipped consistent.

---

## 5. Glossary

| term                  | meaning |
|-----------------------|---------|
| **trusted_starts**    | A frame that survived all of Phase A's 6 cross-verification gates. Its (GPS, heading) tuple is good enough to use as a training anchor. There are 2,177 across 8 videos. |
| **strict**            | After Phase B, only samples whose closed-loop angular error δ < 30° are kept. The "strict pool" is 4,689 of the 6,510 raw teacher outputs. |
| **closed-loop δ**     | `|angle_diff(heading_gt + ACTION_DELTA[parsed_action], first_seg_bearing)|`. If small, the action verb in the answer points the user the right way. Pass thresholds: <30° strict, ≤55° loose. |
| **6 gates**           | step_12 verifier checks: (1) format, (2) sentence count, (3) closed-loop, (4) checkpoint, (5) destination correct, (6) anchor grounded. PASS_strict requires gates 1,2,3,4,6 + δ<30° (gate 5 is reported but not gating). |
| **PASS_LANDMARK / PASS_STREET / FAIL** | Step 10 verdict comparing GPS-near POIs against VLM-seen POIs. Used during Phase A only. |
| **INFERRED_HEADING:** | Mechanical regex slot in the v4b/v4c CoT — the model writes its inferred camera heading on its own line so the verifier can parse it. |
| **hold-out**          | The entire `saturday_morning` walking video (255 v3 samples, 88 trusted_starts frames). Excluded from all training data. |
| **first_seg_bearing** | Absolute compass bearing of the first OSM edge after the user's GPS, computed by way_planner. |
| **ACTION_DELTA**      | `{continue ahead: 0, turn left: -90, turn right: +90, turn around: 180}`. Used in closed-loop math. |
| **Mapillary**         | Crowdsourced street-level photo dataset. We embed 5k of its Zurich images with DINOv2 to use as a visual GPS index. |
| **HMM map matching**  | Newson-Krumm Viterbi snap of a noisy GPS sequence to OSM road geometry. |

---

## 6. Known traps — read before you change anything

1. **Hold-out integrity**. `saturday_morning` is excluded from every
   training file. If you re-derive datasets, double-check no
   `extra_Zurich_looks_STUNNING_on_Saturday_Morning_Switzerl/`
   frame leaks in. The string match is the only protection.

2. **Stale docstrings**. `toolbox/synth/prompts.py` opens with
   "v2 design: model receives no `user_heading`". This is wrong for v3
   (which DOES give heading); v4a/b/c are the no-heading variants.
   The actual SYSTEM_PROMPT in the file is v3-correct.

3. **gate 5 dest_correct** is only ~66% even on the perfectly trained
   C1 LoRA. This is a known prompt-design issue (the multi-turn
   checkpoint name leaks into the destination slot in the answer's
   middle sentence). It does not gate PASS_strict in current
   `eval_lora.py`. See `EXPERIMENT_REPORT.md §3` for analysis.

4. **HQ Mapillary branch (`*_hq` suffix anywhere)** was abandoned. Code
   accepts a `--variant _hq` flag in some places but the data was
   never finalised. Ignore it.

5. **`max_pixels=448²` in training** is critical for fitting on a
   90 GB GPU. If you change it, re-tune everything else.

6. **Teacher VLMs do not run on the same GPU as LoRA training**. 62 GB
   (Gemma) + 50 GB (LoRA) > 90 GB. Run them serially. See
   `docs/TEACHER_DEPLOY.md`.

7. **`build_v4c_rationale.py` calls Anthropic Claude API**. Costs ~$50
   for a full re-run. The output is already shipped at
   `data/cities/zurich/synth_v4c_*.jsonl`; do not regenerate unless
   you specifically need to.

8. **Path assumptions**: All scripts assume you `cd navlm_ss/` first.
   Relative paths everywhere.

---

## 7. Project state as of 2026-04-28

**Done**:
- Phase A pipeline finalised (8 videos → 2,177 trusted_starts).
- Synth_v3 generated and verified (4,689 strict samples).
- v4a / v4b / v4c training sets derived.
- 4 LoRAs trained: lora_zurich_v3, _v4a, _v4b, _v4c.
- 7 conditions evaluated (3 base × 3 prompts + 4 LoRA).
- EXPERIMENT_REPORT.md fills H1-H7 analysis with numbers.
- NeurIPS 2026 paper draft started.
- HQ Mapillary index branch (22.6k images) abandoned — distribution
  shift hurt visual matching.

**Not done / future work**:
- Cross-city evaluation (saturday_morning is same domain as training).
- gate 6 hallucination check was skipped during eval for speed.
- Human evaluation of answer quality.
- Other open items in the original task list (#84 video continuation
  overlay, #97 viewer update, #106 mode-B verifier).

---

## 8. If you only have 60 seconds to brief yourself

Read these in this order:
1. `AGENTS.md` (this file) — already done.
2. `results/EXPERIMENT_REPORT.md` §1 (background) — 2 minutes.
3. `pipeline/PIPELINE.md` outline — 2 minutes scan.

Then you can do anything in §2 above.
