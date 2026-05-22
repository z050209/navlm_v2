# NavLM-SS — self-supervised navigation VLM (snapshot for transfer)

This is a reproducible snapshot of the NavLM project as of 2026-04-28.
NavLM fine-tunes Qwen2.5-VL-7B to give scene-anchored walking
directions, using only YouTube walking-tour videos as training data
(no manual GPS labelling).

> 🤖 **If you are a new agent / engineer with no project context**,
> start with **[AGENTS.md](AGENTS.md)** — it gives a 5-minute orientation
> covering project goal, common tasks, file map, glossary, and known
> traps. Other docs assume you know what `trusted_starts`,
> `closed-loop δ`, and `synth_v3 vs v4a vs v4b vs v4c` mean.

## What's inside

```
navlm_ss/
├── pipeline/                  GPS recovery + verifier (steps 7-12)
│   ├── PIPELINE.md            ★ full step-by-step pipeline doc
│   ├── config.py              video list, paths, thresholds
│   ├── step_07_merge_gps.py   visual_high + bbox sanity
│   ├── step_08_hmm.py         HMM road snapping wrapper
│   ├── step_09_vlm_poi.py     split combined VLM scan per video
│   ├── step_10_vlm_verify.py  PASS_LANDMARK / PASS_STREET / FAIL
│   ├── step_11_trusted.py     trusted_starts filter
│   ├── step_12_closed_loop_verify.py   final 6-gate verifier
│   ├── run_all.py             driver
│   ├── build_v4_datasets.py   v3 → v4a (implicit) + v4b (explicit)
│   └── build_v4c_rationale.py v3 → v4c (Claude rationale)
│
├── toolbox/                   underlying tools (Phase A, OSM, scenery)
│   ├── extract_frames.py      ffmpeg → 1 fps jpg
│   ├── embed_images.py        DINOv2 ViT-L/14 embeddings
│   ├── visual_match_gps.py    nearest-neighbour against Mapillary
│   ├── refine_visual_match.py GPS+compass consensus → high/medium/low
│   ├── ocr_paddle.py          PaddleOCR text detection
│   ├── landmark_match.py      OCR text → ZURICH_LANDMARKS table
│   ├── compute_frame_heading.py  Mapillary neighbour compass average
│   ├── way_planner.py         OSM walking-route planner
│   ├── scenery_pois.py        hand-curated streets/rivers/lakes
│   ├── scan_video_pois_multi.py   Gemma per-frame POI visibility
│   ├── map_match.py           HMM Newson-Krumm Viterbi
│   ├── synth_unified.py       v3 prompt → Gemma teacher → jsonl
│   ├── synth/                 prompts, backends, verifier, sampling
│   └── ...
│
├── scripts/                   training + eval + viz
│   ├── train_lora_cot.py      Qwen2.5-VL LoRA SFT
│   ├── eval_lora.py           base vs LoRA on hold-out, 6 gates
│   ├── plot_eval_comparison.py    bar charts per condition
│   ├── visualize_paths.py     folium 8-video map
│   └── synth_viewer.py        Flask :9000 sample browser
│
├── data/cities/zurich/
│   ├── frames/<video>/        ★ 27,075 jpg frames (8 videos)
│   ├── frame_starts_trusted_all.jsonl   ★ 2,177 frames after Phase A
│   ├── synth_v3_full.jsonl    6,510 raw teacher outputs
│   ├── synth_v3_full_strict.jsonl  4,689 (δ<30°) ★ training pool
│   ├── synth_v3_train.jsonl   4,434 (saturday_morning held out)
│   ├── synth_v3_eval.jsonl    255 hold-out
│   ├── synth_v4a/v4b/v4c train/eval — derived prompt variants
│   ├── landmarks_zurich_osm.json    POI table
│   ├── osm_walking.pkl              OSM walking graph (osmnx)
│   └── _video_poi_multi.jsonl       Gemma POI scan output
│
├── data/cities/mapillary/zurich/    ★ 5k Mapillary reference index
├── data/mapillary/zurich_full/      89k Mapillary index (full version)
│
├── results/
│   ├── EXPERIMENT_REPORT.md   ★ 6-condition writeup (H1-H7)
│   ├── lora_zurich_v3/        ★ EXP-C1 LoRA weights (190 MB adapter)
│   ├── lora_zurich_v4a/       ★ EXP-C2 LoRA weights (implicit)
│   ├── lora_zurich_v4b/       ★ EXP-C3 LoRA weights (explicit)
│   ├── lora_zurich_v4c/       ★ EXP-C4 LoRA weights (Claude rationale)
│   ├── eval_v3_<tag>.json     summary per condition (7 conditions)
│   ├── eval_v3_<tag>.jsonl    per-row eval details
│   └── plot_*.png             4 comparison bar charts
│
├── draft/                     NeurIPS 2026 paper draft
│   ├── main.tex               + introduction.tex / related_work.tex
│   ├── gps_data_pipeline.tex
│   ├── paper.md               full markdown sketch
│   └── neurips_2026.{sty,tex}
│
├── REPRODUCE.md               ★ end-to-end commands to re-run
├── docs/TEACHER_DEPLOY.md     ★ Gemma / Qwen3-VL vLLM deployment guide
├── scripts/serve_teacher.sh   one-command vLLM launcher (gemma | qwen3vl | stop | status)
└── requirements.txt           pip dependencies
```

## Quick start

See `REPRODUCE.md` for full command sequences.

```bash
# 1. environment
pip install -r requirements.txt

# 2. base model (download once, ~16 GB)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir models/Qwen2.5-VL-7B-Instruct

# 3. evaluate the trained LoRA on the hold-out (~30 min on a single GPU)
python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v3_eval.jsonl \
    --lora results/lora_zurich_v3 \
    --tag  lora_v3 \
    --skip-hallucination

# 4. compare base vs LoRA
python scripts/eval_lora.py --compare \
    results/eval_v3_base_v3.json results/eval_v3_lora_v3.json
```

## Headline result (saturday_morning hold-out, 255 frames)

```
                          PASS_strict   closed_loop   median δ
─────────────────────────────────────────────────────────────────
A1  base + heading         38.0%         38.8%        80.7°
A2  base + no heading      23.1%         31.8%        89.8°
A3  base + explicit CoT     9.0%         25.9%        98.7°
C1  LoRA + heading        100.0%        100.0%         8.9°  ← ceiling
C2  LoRA + no heading      46.3%         49.8%        66.1°
C3  LoRA + explicit CoT    52.9%         63.9%        20.8°  ← ⭐
C4  LoRA + Claude CoT      41.6%         48.2%        66.2°
```

The compass-free LoRA (C3) reduces median heading error from 99° (base
zero-shot) to **21°**, narrowing the gap to the with-compass ceiling
(C1, 8.9°). See `results/EXPERIMENT_REPORT.md` for full analysis.

## Notes for whoever picks this up

- **Base model** is Qwen2.5-VL-7B-Instruct. Code expects it at
  `/path/to/models/Qwen2.5-VL-7B-Instruct`. Edit `MODEL_PATH` at the top
  of `scripts/train_lora_cot.py` and `scripts/eval_lora.py`.
- **Teacher VLM** for data generation was Gemma-4-31B served via vLLM at
  `http://localhost:8003/v1`. We ship `scripts/serve_teacher.sh` to
  start it on a single 90 GB GPU; see `docs/TEACHER_DEPLOY.md`.
  v4c rationale data also used Anthropic Claude Sonnet 4.6 —
  set `ANTHROPIC_API_KEY` if regenerating v4c.
- **No GPS labels** anywhere. Phase A recovers GPS purely from visual
  matching against Mapillary + HMM road snapping.
- **Hold-out** is the entire `saturday_morning` walking video (255 v3
  samples). Train sets exclude every frame from this video.
