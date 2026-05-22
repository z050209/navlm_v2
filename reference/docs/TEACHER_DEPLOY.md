# Teacher VLM deployment (Gemma-4-31B and Qwen3-VL-32B via vLLM)

The synth-data pipeline calls a teacher VLM through an OpenAI-compatible
HTTP endpoint. We use **vLLM** to serve open-weights models locally so
neither training data nor evaluation queries leave the machine. Two
teachers are supported:

| model | role | port | pieces of code that hit it |
|-------|------|------|---------------------------|
| **Gemma-4-31B-it** | primary teacher (synth_unified, step 9 POI scan, step 10 hallucination judge) | 8003 | `toolbox/synth/backends.py` (gemma branch), `toolbox/scan_video_pois_multi.py`, `scripts/eval_lora.py` (gate 6) |
| **Qwen3-VL-32B-Instruct** | optional 2nd opinion / stronger hallucination judge | 8004 | gate 6 alternative, future ablations |

Both fit on one ~90 GB GPU in fp16 with reduced context (16k tokens, more
than enough for our prompts).

## 0 — install vLLM

```bash
# In your conda / venv:
pip install "vllm>=0.6.0"

# Verify
vllm --version
```

vLLM >= 0.6 supports both Gemma 3/4 and the Qwen-VL family natively. If
you hit a "Unrecognised model" error, upgrade vLLM.

## 1 — download the model weights

Use HuggingFace to fetch into a local directory (NOT the HF cache, so
you can move it).

```bash
# ~62 GB
huggingface-cli download google/gemma-4-31b-it \
    --local-dir /models/gemma-4-31b-it \
    --local-dir-use-symlinks False

# ~63 GB
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct \
    --local-dir /models/Qwen3-VL-32B-Instruct \
    --local-dir-use-symlinks False
```

(Anywhere on the disk is fine. Just remember the path; you will pass it
to `serve_teacher.sh`.)

## 2 — start the server

We ship a wrapper script in `scripts/serve_teacher.sh`. Edit the two
`*_PATH` variables inside, or pass them in via env:

```bash
# foreground (for testing)
CUDA_VISIBLE_DEVICES=0  \
GEMMA_PATH=/models/gemma-4-31b-it \
bash scripts/serve_teacher.sh gemma

# background (for long-running synth)
CUDA_VISIBLE_DEVICES=0  \
GEMMA_PATH=/models/gemma-4-31b-it \
nohup bash scripts/serve_teacher.sh gemma > /tmp/gemma_serve.log 2>&1 &
disown
```

Wait ~60-90 seconds for the model to load. Verify it's up:

```bash
bash scripts/serve_teacher.sh status
# → should show port 8003 returning a JSON model list
```

## 3 — running both teachers at once

A single 90 GB GPU CANNOT host both Gemma-31B and Qwen3-VL-32B
simultaneously (62 + 63 = 125 GB > 90 GB). If you have only one GPU,
start them on different GPUs:

```bash
# GPU 0 hosts Gemma
CUDA_VISIBLE_DEVICES=0  GEMMA_PATH=...    bash scripts/serve_teacher.sh gemma   &
# GPU 1 hosts Qwen3-VL
CUDA_VISIBLE_DEVICES=1  QWEN3VL_PATH=...  bash scripts/serve_teacher.sh qwen3vl &
```

If you have ONE 90 GB GPU only, run them serially:

```bash
# Phase B: synth generation (need Gemma)
bash scripts/serve_teacher.sh gemma &
SERVE_PID=$!
python toolbox/synth_unified.py ...
kill $SERVE_PID

# Phase D: eval with Qwen3-VL hallucination judge (need Qwen3-VL)
bash scripts/serve_teacher.sh qwen3vl &
SERVE_PID=$!
python scripts/eval_lora.py ... # (without --skip-hallucination)
kill $SERVE_PID
```

## 4 — pointing the navlm code at the server

The synth-data pipeline calls the OpenAI-compatible endpoint at
`http://localhost:<port>/v1/chat/completions`. The expected endpoint
is configured per call:

| script | flag | default |
|--------|------|---------|
| `toolbox/synth_unified.py` | `--vllm-url http://localhost:8003/v1` `--model google/gemma-4-31b-it` | port 8003, Gemma |
| `toolbox/scan_video_pois_multi.py` | `--vllm-url ...` `--model ...` | port 8003, Gemma |
| `scripts/eval_lora.py` (gate 6) | hard-coded constants `GEMMA_URL`, `GEMMA_MODEL` | port 8003, Gemma |

To switch the verifier to Qwen3-VL-32B, edit
`scripts/eval_lora.py`:

```python
GEMMA_URL   = "http://localhost:8004/v1"
GEMMA_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
```

(or a future flag — currently constants).

## 5 — memory tuning for smaller GPUs

If you have less than 80 GB:

```bash
# 8-bit weights (vLLM supports awq / gptq / bnb)
vllm serve $GEMMA_PATH \
    --quantization bitsandbytes --load-format bitsandbytes \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85

# Or aggressive context shortening (synth prompts are ~3k tokens)
vllm serve $GEMMA_PATH \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95
```

For 24 GB cards, you really need tensor-parallel across multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $GEMMA_PATH \
    --tensor-parallel-size 4 \
    --max-model-len 16384
```

## 6 — quick smoke test

After starting Gemma, hit it with a tiny request:

```bash
curl -sS http://localhost:8003/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "google/gemma-4-31b-it",
      "messages": [{"role":"user","content":"Reply with exactly OK"}],
      "max_tokens": 5
    }' | python3 -m json.tool
```

Expect a JSON response with `choices[0].message.content == "OK"`.

If you see `connection refused`, the model is still loading — wait
another minute. If `404 Not Found`, the served-model-name doesn't
match what the script asks for.

## 7 — common failure modes

| symptom | cause | fix |
|---------|-------|-----|
| `out of memory` at startup | model weights + KV cache > GPU | lower `--max-model-len`, raise `--gpu-memory-utilization` cautiously |
| `connection refused` from synth | server not yet ready | wait 60–120 s for first load; check log |
| `model 'google/gemma-4-31b-it' not found` | served-model-name mismatch | check `--served-model-name` matches what synth_unified passes |
| OOM mid-run on long prompts | `max_pixels` of teacher's vision encoder too high | edit `scripts/serve_teacher.sh` and pass `--limit-mm-per-prompt '{"image":1}'` |
| training and serving on same GPU collide | 50 + 70 > 90 GB | run them serially, not concurrently |

## 8 — when you don't need the teacher

You can skip teacher deployment entirely if you only want to:

1. **Evaluate a shipped LoRA** on the held-out set
   → use `--skip-hallucination` flag on `eval_lora.py`. Skips gate 6.
2. **Re-train a LoRA from already-generated synth jsonl**
   → training only uses `train_lora_cot.py` which doesn't call any teacher.
3. **Browse data** in the `:9000` viewer
   → reads JSON files only.

Teacher VLMs are required only when **regenerating synth data** (Phase B)
or when scoring **gate 6 hallucination** at eval time.
