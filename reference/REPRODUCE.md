# Reproducing NavLM-SS results

End-to-end command sequence for the new server. Assumes a fresh box
with CUDA 12+ and Python 3.10+.

## GPU resource requirements

A single 80–90 GB GPU (A100 80 GB, H100, L20X 141 GB) is sufficient
for everything in this repo. Tasks must run **serially** because some
steps each need the full GPU:

| step | peak GPU memory | notes |
|------|----------------|-------|
| L1 — evaluate shipped LoRA | ~25 GB | inference only |
| L2 — re-train LoRA | ~50 GB | `max_pixels = 448²` |
| L3 — re-generate synth via Gemma | ~70 GB | Gemma-4-31B served |
| L4 — Phase A re-run | ~70 GB | Gemma POI scan + DINOv2 + PaddleOCR (serial) |

You CANNOT train LoRA *and* serve Gemma teacher simultaneously on a
single 90 GB card (50 + 70 = 120 GB > 90 GB). Run synth generation
first, kill Gemma, then train.

External downloads required:

| asset | size | when needed |
|-------|------|------------|
| Qwen2.5-VL-7B-Instruct weights | ~16 GB | always |
| Gemma-4-31B-it weights | ~62 GB | only for L3+ (re-gen synth) |
| Anthropic API key | — | only for re-generating v4c |
| raw .mp4 videos | ~3 GB | only for L4 (Phase A from scratch) |

## 0 — environment

```bash
cd /path/to/navlm_ss
pip install -r requirements.txt

# huggingface base model (~16 GB) — required for ALL paths
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir models/Qwen2.5-VL-7B-Instruct

# Update MODEL_PATH in scripts/train_lora_cot.py and scripts/eval_lora.py
# to point at the local copy.

# OPTIONAL: teacher VLM weights (~62 GB Gemma, ~63 GB Qwen3-VL).
# Only needed for L3+ (regenerating synth) or for gate 6 hallucination
# during eval. See docs/TEACHER_DEPLOY.md for deployment details.
huggingface-cli download google/gemma-4-31b-it \
    --local-dir models/gemma-4-31b-it
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct \
    --local-dir models/Qwen3-VL-32B-Instruct
```

## 0.5 — teacher VLM serving (only for L3+ / gate 6)

`toolbox/synth_unified.py` and the `gate 6 anchor_grounded` check both
hit an OpenAI-compatible HTTP endpoint at `http://localhost:8003/v1`.
This is fed by Gemma-4-31B (or optionally Qwen3-VL-32B) served via vLLM.

```bash
# Start Gemma teacher on a free GPU (~62 GB model, fits on one 90 GB card)
CUDA_VISIBLE_DEVICES=0  GEMMA_PATH=models/gemma-4-31b-it \
    nohup bash scripts/serve_teacher.sh gemma > /tmp/gemma.log 2>&1 &
disown

# Wait ~90 s for it to load
bash scripts/serve_teacher.sh status
# → port 8003 returns the model JSON when ready

# Stop later
bash scripts/serve_teacher.sh stop
```

⚠️ A single 90 GB GPU **cannot** host Gemma teacher AND train LoRA
simultaneously (62 + 50 ≈ 112 GB > 90 GB). Run them serially. See
`docs/TEACHER_DEPLOY.md` for full deployment guide, multi-GPU options,
quantization, and switching to Qwen3-VL-32B.

If you only want to **evaluate the shipped LoRAs** (L1) or **re-train
on the cached synth jsonl** (L2), you don't need the teacher running at
all — pass `--skip-hallucination` to `eval_lora.py`.

## 1 — minimum: just evaluate the shipped LoRA

You already have:
- `data/cities/zurich/synth_v3_eval.jsonl`           (255 hold-out)
- `data/cities/zurich/frames/<video>/*.jpg`          (27k frames)
- `results/lora_zurich_v3/adapter_model.safetensors` (190 MB adapter)

```bash
# ~30 min on one GPU
CUDA_VISIBLE_DEVICES=0 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v3_eval.jsonl \
    --lora results/lora_zurich_v3 \
    --tag  lora_v3_repro \
    --skip-hallucination
```

Expected: `pass_strict_30 ≈ 100%` on closed-loop, matching
`results/eval_v3_lora_v3.json`.

## 2 — re-train from synth data (skip Phase A)

You already have `synth_v3_train.jsonl` (4,434) and `synth_v3_eval.jsonl`
(255). LoRA training takes ~3 hours on one L20X / A100 80 GB.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_cot.py \
    --train  data/cities/zurich/synth_v3_train.jsonl \
    --val    data/cities/zurich/synth_v3_eval.jsonl \
    --output results/lora_zurich_v3_repro \
    --epochs 2 --batch_size 1 --grad_accum 8 \
    --lr 2e-4 --lora_r 16 --lora_alpha 32

# evaluate
CUDA_VISIBLE_DEVICES=0 python scripts/eval_lora.py \
    --eval data/cities/zurich/synth_v3_eval.jsonl \
    --lora results/lora_zurich_v3_repro \
    --tag  lora_v3_repro
```

## 3 — re-generate synth data from trusted_starts (skip Phase A → use cached)

Requires a teacher VLM server. Deploy Gemma first:

```bash
# Start Gemma on GPU 0 (see docs/TEACHER_DEPLOY.md)
CUDA_VISIBLE_DEVICES=0  GEMMA_PATH=models/gemma-4-31b-it \
    nohup bash scripts/serve_teacher.sh gemma > /tmp/gemma.log 2>&1 &
disown
sleep 90  # wait for load
```

Or set `--backend openai`/`anthropic` and configure `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` to use a cloud teacher.

```bash
# ~6 hours on Gemma at port 8003
python toolbox/synth_unified.py \
    --starts data/cities/zurich/frame_starts_trusted_all.jsonl \
    --out    data/cities/zurich/synth_v3_full_repro.jsonl \
    --n-dest-per-frame 3 \
    --skip-visual-verify \
    --backend gemma

# verify + filter to strict pool
python -m pipeline.step_12_closed_loop_verify \
    --in  data/cities/zurich/synth_v3_full_repro.jsonl \
    --out data/cities/zurich/synth_v3_full_verified_repro.jsonl

# (the strict-filter helper is a few lines of Python; see report.py
#  for the pattern, or extract via:)
python3 -c "
import json
with open('data/cities/zurich/synth_v3_full_verified_repro.jsonl') as fin, \
     open('data/cities/zurich/synth_v3_strict_repro.jsonl', 'w') as fout:
    for ln in fin:
        r = json.loads(ln)
        if r['_meta'].get('passes_strict'):  # set by step_12 in --strict mode
            fout.write(ln)
"
```

## 4 — full Phase A from raw videos (longest path)

Required only if you want to re-run GPS recovery from raw frames.
Assumes you have the source videos somewhere, and `data/cities/mapillary/zurich/`
is populated (it is in this snapshot).

```bash
# Step 1-2 — extract frames + DINOv2 embeddings (per video)
python toolbox/extract_frames.py \
    --video data/raw/<video>.mp4 \
    --out   data/cities/zurich/frames/<video>
CUDA_VISIBLE_DEVICES=0 python toolbox/embed_images.py \
    --frames data/cities/zurich/frames/<video> \
    --out    data/cities/zurich/frames/<video>_embeddings.npz

# Step 3-4 — visual match + refine (per video)
python toolbox/visual_match_gps.py \
    --query-embeds data/cities/zurich/frames/<video>_embeddings.npz \
    --mly-embeds   data/cities/mapillary/zurich/embeddings.npz \
    --mly-meta     data/cities/mapillary/zurich/meta.jsonl \
    --out          data/cities/zurich/<video>_frame_gps.jsonl
python toolbox/refine_visual_match.py \
    --frame-gps data/cities/zurich/<video>_frame_gps.jsonl \
    --mly-meta  data/cities/mapillary/zurich/meta.jsonl \
    --out       data/cities/zurich/<video>_frame_gps_refined.jsonl

# Step 5-6 — OCR (per video)
python toolbox/ocr_paddle.py \
    --frames data/cities/zurich/frames/<video> \
    --out    data/cities/zurich/<video>_frame_ocr.jsonl
python toolbox/landmark_match.py \
    --ocr  data/cities/zurich/<video>_frame_ocr.jsonl \
    --out  data/cities/zurich/<video>_frame_gps_ocr.jsonl

# Step 7-11 — strict pipeline (driver does all 8 videos)
python -m pipeline.run_all --from-step 7
# → data/cities/zurich/frame_starts_trusted_all.jsonl
```

## 5 — replicate full 6-condition experiment

Requires GPUs and ~12 hours total.

```bash
# Train all 4 LoRAs (parallel on 4 GPUs would be ideal)
for variant in v3 v4a v4b v4c; do
    CUDA_VISIBLE_DEVICES=$N python scripts/train_lora_cot.py \
        --train  data/cities/zurich/synth_${variant}_train.jsonl \
        --val    data/cities/zurich/synth_${variant}_eval.jsonl \
        --output results/lora_zurich_${variant}_repro \
        --epochs 2 --batch_size 1 --grad_accum 8 \
        --lr 2e-4 --lora_r 16 --lora_alpha 32
done

# Evaluate all 7 conditions
for variant in v3 v4a v4b; do
    # base
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_lora.py \
        --eval data/cities/zurich/synth_${variant}_eval.jsonl \
        --tag  base_${variant} --skip-hallucination
done
for variant in v3 v4a v4b v4c; do
    # LoRA
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_lora.py \
        --eval data/cities/zurich/synth_${variant}_eval.jsonl \
        --lora results/lora_zurich_${variant}_repro \
        --tag  lora_${variant}_repro --skip-hallucination
done

# Plot all 7 bars
python scripts/plot_eval_comparison.py
```

## File-size cheatsheet (what's where)

| dir | size | regenerable? |
|-----|------|--------------|
| `data/cities/zurich/frames/` | ~13 GB | yes (extract from videos) |
| `data/mapillary/zurich_full/` | ~3.6 GB | yes (re-fetch from Mapillary) |
| `data/cities/mapillary/zurich/` | ~? | yes |
| `results/lora_zurich_*/` | ~800 MB | yes (~3 h GPU each) |
| `data/cities/zurich/synth_*.jsonl` | ~250 MB | yes (~6 h Gemma each) |
| `data/cities/zurich/frame_starts_trusted_all.jsonl` | 1.3 MB | yes (Phase A: 4 h) |
| code (pipeline + toolbox + scripts) | <1 MB | — |
| `draft/` | <1 MB | — |

## Troubleshooting

- **Image-pixel cap mismatch**: training uses `max_pixels = 448²`. If
  you hit OOM at training, lower it further. See `train_lora_cot.py`.
- **Gemma server**: `synth_unified.py --backend gemma` expects vLLM at
  `localhost:8003/v1` serving `google/gemma-4-31b-it`. Adjust
  `--vllm-url` and `--model` flags as needed.
- **Hold-out integrity**: `saturday_morning` is excluded from training
  by `pipeline/build_v4_datasets.py` and `synth_v3_train.jsonl` was
  built with the same hold-out. If you re-derive datasets, double-check
  no `saturday_morning` frames leak in.
- **Path assumptions**: code expects relative paths from the repo root.
  Run all commands with `pwd == navlm_ss/`.
