# AI-Usage Attribution — NavLM v2 / CS231n Final Project

**Author**: Yi (z050209) — single-author project
**Course**: Stanford CS231n, Spring 2026
**Submission date**: 2026-06-05

This document is the required attribution record for AI-assistance
used in this project. It catalogues:

1. Every AI-generated artifact (code, prompts, manual sections,
   figures), with the human-vs-AI division of work.
2. Pointers into the full conversation transcript, grouped by
   pipeline component.
3. The location of the raw transcript JSONL evidence file.
4. Per-artifact attribution.

---

## 0. AI tools used

| Tool | Vendor / model | Role | Where used |
|---|---|---|---|
| **Claude Code** | Anthropic, `claude-opus-4-7` (interactive CLI) | Software-engineering assistance: writing Python scripts, debugging Modal/PowerShell issues, generating matplotlib figures, drafting documentation, proofreading | Used throughout — see §2 for per-component breakdown |
| **Gemini Pro 2.5** | Google DeepMind (Vertex AI) | **Method-level use, not writing aid**: teacher model for SFT data annotation (3-pass per variant). Fully documented in material.md §3.3 as part of the experimental method. | `src/a2_annotate.py`, 3 GCP service-account projects |

**Not used**: ChatGPT/GPT-4, Copilot, Gemini for writing/coding,
DeepSeek, Llama, any other AI service for code or text generation.

---

## 1. Raw transcript evidence

The full Claude Code conversation transcript is archived at:

- **Local**: `C:\Users\z0502\.claude\projects\G--My-Drive-cs231n-project-cs231n\b5bcb9d3-30bb-4725-93c3-a3caf084e779.jsonl`
- **Backup (Google Drive)**: `G:\My Drive\cs231n\project\claude_chat_20260603_201550_b5bcb9d3.jsonl`
- **Format**: JSONL — one JSON message per line (user prompts, AI
  responses, tool calls, tool results)
- **Size**: ~38.6 MB (covers ~10 days of work, ~2026-05-26 to 2026-06-04)
- **Session ID**: `b5bcb9d3-30bb-4725-93c3-a3caf084e779`

This JSONL contains every prompt the human author wrote and every
response (including all generated code) Claude Code returned. It is
the canonical evidence file for AI attribution. If the grading TA
needs to verify any specific code-generation moment, it can be
located in this transcript by searching for the relevant function
name or filename.

---

## 2. AI-generated artifacts — per pipeline component

Each subsection below covers (a) what the component is, (b) what was
AI-generated vs human-decided, and (c) where in the transcript the
relevant generation happened. Components are listed in the order the
user requested.

### 2.1 Data downloading & preprocessing

**What it is**: Source-data acquisition — the Zurich walking-tour
videos, the StreetView panorama grid, the DINOv2 model weights, and
the OSM walking-graph file for Zurich.

**Status**: All source data was downloaded/prepared in an EARLIER
project (Attempt 1) before this conversation began. Specifically:
- `gps_recovery_full.jsonl`, `poi_scan.jsonl`,
  `poi_scan_cos0.75.jsonl` — produced by Attempt-1 scripts
  (`src/gps_recovery.py`, `src/poi_scan.py`, `src/_vlm_test.py`).
- `osm_walking.pkl` — produced by `src/build_walking_graph.py`
  (Attempt 1).
- `data/cities/zurich/frames/*.jpg` — extracted from raw MP4s by
  Attempt-1 video processing.

**This conversation's contribution**: NONE on raw download. We
treated these files as read-only inputs to the Attempt-2 pipeline.

**AI-generated code**: NONE.

### 2.2 HMM map-matching

**What it is**: Hidden-Markov-model snapping of noisy DINOv2-recovered
GPS to the OSM walking graph, producing `gps_snapped` +
`segment_id` + `segment_bearing` per frame
(Newson & Krumm, 2009).

**Status**: Implementation `src/road_snap.py` is from Attempt 1 —
NOT AI-generated for this project. In Attempt 2 we re-ran it on a
wider cohort:
```bash
python -m src.road_snap --tier 1 --top-pois 0 \
    --output data/cities/zurich/a2/road_snapped_a2.jsonl
```

**This conversation's contribution**:
- Decision to re-run with `--top-pois 0` (no POI filter) to cover all
  2,470 VLM-accepted frames (Attempt 1 had filtered to 30 POIs).
- The `src/a2_destination_targets.py` and `src/a2_route.py` scripts
  use HMM-snapped GPS via `gps_snapped` field — AI-assisted with
  human-directed design.

**AI-generated code**:
- `src/a2_destination_targets.py` (203 lines) — AI-assisted, human-
  directed. Pyproj-based UTM transformer for nearest-node queries
  was an AI-suggested fix after a 57 km snap-offset bug.
- `src/a2_route.py` (~250 lines) — AI-assisted, human-directed.

**Transcript pointer**: Search for `_make_projector(G)` and "UTM
projection error" in the JSONL.

### 2.3 OSM walking-graph routing (multi-target nearest-of-N)

**What it is**: Per-(frame, destination) shortest-path routing via
`nx.shortest_path` on the UTM-projected OSM walking graph. For
5 "long-feature" attractions (Lake Zurich, Limmat, Bahnhofstrasse,
Niederdorfstrasse, Limmatquai), the destination is a LIST of
candidate nodes and we pick the network-nearest (not straight-line-
nearest) by computing Dijkstra to every candidate and taking the
min.

**This conversation's contribution**: SIGNIFICANT.
- The decision to do multi-target nearest-of-N (vs single-target
  centroid) was discussed and human-finalised — see the transcript
  conversation around "Limmat river — 313 candidate nodes, take the
  shortest-walking-distance one".
- The Dijkstra-per-candidate loop was AI-implemented in
  `src/a2_route.py:139-165`.

**AI-generated code**:
- `src/a2_route.py:139-165` (`shortest_path_to_target`) — AI-implemented
  per a verbal specification.
- The `_make_projector(G)` helper that handles UTM projection — AI-
  implemented after the human author identified that
  `osm_walking.pkl` is UTM-projected and `ox.distance.nearest_nodes`
  was returning wrong results.

**Verification**: Human author manually inspected the 313 Limmat-
river candidates' geographic distribution to verify multi-target
made sense.

### 2.4 DINOv2 matching at cos ≥ 0.75 (per-frame coincidence)

**What it is**: STEP 3 of the data pipeline — per-frame coincidence
match between GPS-side OSM candidates (`GPS_GEO.jsonl`) and VLM-side
candidates (`VLM_GEO.jsonl`), filtered to DINOv2 cosine ≥ 0.75. A
frame is "matched" iff at least one name from the GPS side coincides
with at least one from the VLM side (exact / substring / word-share
match).

**Implementation**: `src/a2_step3_gps_vlm_geo.py`.

**This conversation's contribution**:
- Decision to use cos ≥ 0.75 as the cutoff: human (carried over from
  Attempt 1).
- Decision to allow 3 coincidence types (exact / substring /
  word-share): human-decided, AI-implemented.
- `best_level` field hierarchy (attraction > landmark > poi):
  human-designed, AI-implemented.

**AI-generated code**:
- `src/a2_step3_gps_vlm_geo.py` (~250 lines) — AI-implemented per
  verbal specification. Result: 1,219 matched frames out of 4,158 at
  cos ≥ 0.75.
- `src/a2_step1_gps_geo.py` and `src/a2_step2_vlm_geo.py` —
  AI-implemented (the GPS-side and VLM-side prep that STEP 3 joins).

**Verification**: Human author manually inspected 30 random matched
frames via `viz/a2_vlmagreed.html` (also AI-generated) to verify the
DINOv2 nearest-pano was visually correct.

### 2.5 Instruction tuning — 3-pass Gemini Pro 2.5 annotation

**What it is**: Each of 3,657 (frame, destination) pairs is
annotated 3 times by Gemini Pro 2.5 — once per input variant
(given / derived / implicit). The teacher always sees the GT heading;
the student's prompt is variant-specific. Three GCP service-account
projects parallelise the calls.

**This conversation's contribution**: SIGNIFICANT. The 3-variant
input-asymmetric design was the central scientific contribution of
this project. Several design iterations are visible in the
transcript:

- The "Option D" design (separate annotation passes, teacher always
  has heading) was reached after rejecting "Option B" (single-pass
  multi-section CoT) due to information-leak concerns —
  human-decided, with extensive discussion in the transcript.
- The "interactive guide" answer style ("Can you see X?",
  "Notice...") was added after explicit human direction not to name
  the destination unless it's also in the visible-landmarks list.

**AI-generated code**:
- `src/a2_annotate.py` (~450 lines) — AI-implemented per human-
  directed prompt-design specification. Key functions:
  - `SYSTEM_PROMPT_COMMON_HEAD` — Zurich orientation facts, the
    `<answer>` format rules, GOOD/AVOID examples. Human-drafted,
    AI-polished.
  - `THINKING_RULE[variant]` — the per-variant CoT templates
    (4-step for derived, 3-step for implicit). Human-designed,
    AI-formalised.
  - `build_teacher_prompt(variant)` / `build_student_prompt(variant)` —
    AI-implemented per the input-asymmetry spec.
- `src/gemini_api.py` modifications (per-process auth, parallel-safe
  service-account key loading) — AI-implemented after human-reported
  a Vertex AI 401 error when running 3 parallel passes.
- `_setup_3_gcp_projects.sh`, `_launch_3_annotation_passes.sh` —
  AI-generated shell wrappers, human-edited (BILLING_ID).

**Decisions that were 100% human**:
- "Use Gemini Pro 2.5 as the teacher, not GPT-4 or Claude" — cost +
  multimodal coverage rationale.
- "Always show the teacher the heading, hide it from the student for
  derived/implicit" — the central design choice that makes this
  paper a paper.
- "Add the `--only-pass` filter to drop the teacher's turn-around-
  bias mis-labels" — derived from human-led inspection of the
  confusion matrix in DEV_MANUAL §18.

### 2.6 LoRA supervised fine-tuning

**What it is**: Qwen 2.5 VL 7B (4-bit NF4 base) + LoRA adapters
trained on the 3-pass annotations. Rank sweep r ∈ {4, 8, 16}, epoch
sweep {3, 5}, run on Modal A100-80GB.

**This conversation's contribution**: Both design and implementation.

**AI-generated code**:
- `src/a2_train_modal.py` (~210 lines) — AI-implemented end-to-end.
  Includes:
  - 4-bit NF4 base + BF16 LoRA setup.
  - Custom `collate()` with **assistant-token-only loss masking**
    (the most important methodological detail — added after a
    human-directed overfit test exposed the dominance of
    system+user-token loss).
  - `EarlyStoppingCallback(patience=2)` + `load_best_model_at_end`.
  - `--resume-adapter` flag for continue-training (added late after
    the human author asked "can the code do that?").

**Decisions that were 100% human**:
- "Sweep LoRA rank r ∈ {4, 8, 16}" — the rank-saturation question.
- "Mask the loss to assistant tokens only" — after the human author
  noticed the val_loss curve looked artificially smooth.
- "Add early-stop + load-best-model" — after the human author
  observed e3 might not be the optimal epoch count.
- "Extend by 2 more epochs via warm restart instead of running fresh
  5-epoch training" — to save compute.

**Bug fixes (AI-debugged, human-directed)**:
- Adapter paths were missing the `/ckpts/` mount prefix — caused
  `PeftModel.from_pretrained` to fail with `Can't find
  'adapter_config.json'`. Fixed by editing `DEFAULT_ADAPTER` dict.
- `--lora-r` CLI flag was missing from `local_entrypoint`. Fixed by
  adding it.
- Multi-rank evals were overwriting each other (`run_id/<condition>/`
  collision). Fixed by adding rank suffix to output dir.

### 2.7 Prompt engineering (for both teacher and student)

**What it is**: The actual text strings sent to Gemini Pro 2.5
(teacher) and Qwen 2.5 VL 7B (student). Three variant-specific
system prompts + three variant-specific user-prompt templates.

**This conversation's contribution**: All prompts were collaboratively
designed and iteratively refined.

- **SYSTEM_PROMPT_COMMON_HEAD** (~30 lines, 8 Zurich orientation
  facts + answer-format rules + GOOD/AVOID examples): human-drafted,
  AI-polished. Located in `src/a2_annotate.py:52-82`.
- **THINKING_RULE["given"]** (1-2 sentences using heading + bearing):
  fully human-authored, AI-edited for clarity.
- **THINKING_RULE["derived"]** (4-step CoT template): designed jointly.
  Human specified the steps (visual cues → geography → estimated
  heading → verb); AI wrote the example/template prose.
- **THINKING_RULE["implicit"]** (3-step visual-only template): same
  process as derived.

**Prompt iterations visible in transcript**:
- Multiple rounds of "the answer should be more interactive" → added
  `"Can you see X?"`, `"Notice..."`, `"Look at..."` patterns.
- "Don't name destination unless visible" → AVOID rule added.
- "Use COT inside thinking, not answer" → enforced via template.

**Final prompts**: Documented verbatim in `DEV_MANUAL_v2.md §18`
"Annotation prompt v2 — three independent teacher passes".

### 2.8 Evaluation pipeline + scoring

**What it is**: Per-condition Modal inference (`a2_eval_modal.py`),
local scoring (`a2_score.py`) producing the 4 metrics.

**AI-generated code**:
- `src/a2_eval_modal.py` (~220 lines).
- `src/a2_score.py` (~180 lines).
- `src/a2_to_sft.py` (~150 lines).
- `src/a2_figures.py` (~300 lines) — matplotlib figures for the
  report.
- `src/a2_extract_examples.py` (~125 lines) — qualitative example
  extractor.

**Decisions that were 100% human**:
- The 4-metric design (PASS = format ∧ direction; heading-inference
  only for derived).
- The 22.5° tolerance for heading inference (half of 45°, the
  angular boundary between verb classes).
- The decision to add bimodality analysis at multiple tolerances
  (within 5° / 22.5° / 45° / mean / median) after spotting the
  within-5° ≈ within-22.5° pattern.

### 2.9 Visualizations (HTML QC outputs)

**What it is**: Five interactive HTML pages for visual quality
control of the data pipeline.

**AI-generated code**:
- `src/a2_viz_matched.py` → `viz/a2_vlmagreed.html` (30 random
  matched frames with compass crops, heading badges).
- `src/a2_viz_map.py` → `viz/a2_mapped_GPS_spot.html` (folium map
  of 89 matched panos).
- `src/a2_viz_thin.py` → `viz/a2_thin_attractions.html` (thin-cohort
  cases).
- `src/a2_viz_route_gt.py` → `viz/a2_route_gt.html` (route + GT verb
  per frame).
- `src/a2_viz_sft.py` → `viz/a2_viz_sft.html` (SFT-data QC viewer).

All AI-implemented; the design (what to visualise) was human-
directed; the output HTML was human-reviewed before being kept.

### 2.10 Documentation (DEV_MANUAL_v2.md and report)

**`DEV_MANUAL_v2.md`** (3,353 lines, 26 sections) — AI-drafted from
the actual code and human-directed conversation. Every measured
number in the manual came from running the code; the AI only
formatted and contextualised. Human-reviewed and frequently edited
throughout the project.

**`docs/material.md`** (current file's sibling) — AI-drafted from the
manual content, restructured to match the CS231n guidelines.
Mathematical formalisation in §1 and §3 was AI-authored from
verbal/code descriptions of the algorithm. Citations were AI-
selected from the canonical literature for each topic and human-
verified.

**`docs/AI_USAGE_ATTRIBUTION.md`** (this file) — AI-drafted by
explicit human instruction to document every AI-generated artifact;
human-reviewed.

---

## 3. Plans, prompts, and key conversation moments

A condensed index of decisions visible in the transcript JSONL
(grouped by pipeline component, in chronological order within each
component):

**Data pipeline**:
- Decision to filter destinations to 21 famous attractions instead
  of top-30 OSM POIs — human-led, rationalized via Zurich Tourism
  reference.
- Decision to use cos ≥ 0.75 as the DINOv2 coincidence threshold —
  carried over from Attempt 1.
- UTM-projection bug fix (57 km snap offsets → ~10 m) — AI debugged
  after human pointed out the impossible snap distances.

**Annotation**:
- Rejection of single-pass multi-section CoT (information leak from
  the visible CoT) in favour of 3 parallel passes — human-led.
- Decision to always show teacher the heading — human-led, central.
- Vertex-AI 401 / per-process auth refactor — AI debugged.
- 3 GCP project setup for parallel quota — AI-implemented shell
  wrapper; human filled in BILLING_ID.

**Training**:
- Loss-masking decision (assistant tokens only) — human-noticed val
  loss curve anomaly; AI implemented the mask.
- Early-stop + best-model-load decision — human-directed after first
  smoke run.
- `--resume-adapter` feature for continue-training — added after
  human asked "does the code allow continue training?".
- The rank sweep r ∈ {4, 8, 16} — human-decided, AI-launched.

**Evaluation**:
- 4-metric design — human-led.
- Adapter-path / overwrite bug discovery — AI investigation triggered
  by human noticing that re-runs gave identical numbers.
- The bimodal heading-prediction finding — AI computed the
  within-5° / within-22.5° / mean / median table; human noticed the
  near-equality of within-5° and within-22.5° and named the finding
  "bimodal".

**Report & figures**:
- Figure design (3 figures: rank-saturation, zs-vs-trained, heading
  scatter) — jointly designed.
- Report structure (CS231n 8-section template) — human-supplied
  guidelines; AI-restructured the manual content to fit.

---

## 4. Per-file attribution summary

```
File                                  AI-generated   Human-directed
─────────────────────────────────────────────────────────────────
src/a2_annotate.py                    yes            design/prompts
src/a2_attraction_slots.py            yes            curation list
src/a2_destination_targets.py         yes            algorithm
src/a2_eval_modal.py                  yes            metric design
src/a2_extract_examples.py            yes            -
src/a2_figures.py                     yes            figure design
src/a2_heading_v2.py                  yes            gap-tier rule
src/a2_route.py                       yes            multi-target alg
src/a2_score.py                       yes            4-metric design
src/a2_step1_gps_geo.py               yes            radius choice
src/a2_step2_vlm_geo.py               yes            alias table
src/a2_step3_gps_vlm_geo.py           yes            coincidence rules
src/a2_target_frames.py               yes            -
src/a2_to_sft.py                      yes            --only-pass
src/a2_train_modal.py                 yes            loss-mask, rank sweep
src/a2_viz_*.py (5 files)             yes            QC content
src/gemini_api.py (modifications)     yes            auth refactor
viz/a2_*.html (5 files)               yes (rendered) -
docs/material.md                        yes (drafted)  results, claims
DEV_MANUAL_v2.md                      yes (drafted)  all decisions
docs/figures/*.png                    yes (generated)figure design
docs/qualitative_examples.md          yes (extracted)-
_launch_3_annotation_passes.sh        yes            -
_setup_3_gcp_projects.sh              yes            BILLING_ID filled
.gitignore                            yes            keys/ entry
```

`src/road_snap.py`, `src/gps_recovery.py`, `src/poi_scan.py`,
`src/pois.py`, `src/build_walking_graph.py`, `src/dinov2_match.py`,
`src/streetview.py`, and all upstream input data (frames, OSM, DINOv2
weights) are from the prior Attempt-1 project and were NOT modified
in this submission.

---

## 5. Honor-code statement

The author affirms that:

1. All scientific decisions (experimental design, ablation matrix,
   the 3-variant teacher-student input-asymmetry design, the
   loss-masking choice, the interpretation of results, the bimodal-
   heading finding) were directed by the human author.
2. All code in `src/a2_*.py` was generated with AI software-
   engineering assistance from Claude Code and was reviewed,
   tested, and integrated by the human author.
3. Gemini Pro 2.5's role as the SFT teacher model is a methodological
   choice (analogous to using a pretrained backbone) and is
   documented in material.md §3.3 and DEV_MANUAL §18, not as a writing
   aid.
4. No AI-generated text was passed off as the human author's
   independent prose without disclosure; the material.md was AI-
   drafted, human-reviewed, and revised before submission.
5. The full conversation transcript (38.6 MB JSONL) is preserved as
   evidence at the paths in §1 above, available on request.
6. This project is submitted exclusively to CS231n and is not
   double-counted for any other class.

— Yi (z050209), 2026-06-04
