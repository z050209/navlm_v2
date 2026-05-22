#!/bin/bash
# Serve a teacher VLM via vLLM for synth_unified.py / step_10 / step_12 hallucination check.
#
# Usage:
#   bash scripts/serve_teacher.sh gemma     # Gemma-4-31B-it on port 8003
#   bash scripts/serve_teacher.sh qwen3vl   # Qwen3-VL-32B-Instruct on port 8004
#   bash scripts/serve_teacher.sh stop      # kill any vllm serve owned by this user
#
# Default GPU: CUDA_VISIBLE_DEVICES=0 (single ~90 GB card).
# Both 31B and 32B models fit in fp16 on one 90 GB GPU with shorter context.
#
# Prereqs:
#   pip install vllm>=0.6.0
#   (Models must be pre-downloaded; see paths below.)

set -e

# ────────────────────── config ──────────────────────
# Adjust these to wherever you've put the model weights:
GEMMA_PATH="${GEMMA_PATH:-/path/to/gemma-4-31b-it}"
QWEN3VL_PATH="${QWEN3VL_PATH:-/path/to/Qwen3-VL-32B-Instruct}"

GEMMA_PORT=8003
QWEN3VL_PORT=8004

# Single-GPU deploy assumes ~90 GB available. Drop max-model-len if you
# need more headroom; synth_unified only needs ~8k tokens per call.
SHORT_CONTEXT=16384
GPU_UTIL=0.92

CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

case "$1" in
  gemma)
    if [ ! -d "$GEMMA_PATH" ]; then
        echo "ERROR: GEMMA_PATH not found: $GEMMA_PATH"
        echo "Set it via:  export GEMMA_PATH=/your/path/to/gemma-4-31b-it"
        exit 1
    fi
    echo "[serve] Gemma-4-31B-it on port $GEMMA_PORT, GPU $CUDA_DEV"
    CUDA_VISIBLE_DEVICES=$CUDA_DEV vllm serve "$GEMMA_PATH" \
        --host 0.0.0.0 --port $GEMMA_PORT \
        --served-model-name google/gemma-4-31b-it \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization $GPU_UTIL \
        --max-model-len $SHORT_CONTEXT \
        --dtype bfloat16
    ;;

  gemma-tp4)
    # Original 4-GPU deploy (kept for reference if you have multi-GPU)
    if [ ! -d "$GEMMA_PATH" ]; then
        echo "ERROR: GEMMA_PATH not found"; exit 1
    fi
    CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$GEMMA_PATH" \
        --host 0.0.0.0 --port $GEMMA_PORT \
        --served-model-name google/gemma-4-31b-it \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 262144 \
        --limit-mm-per-prompt '{"video":2}'
    ;;

  qwen3vl)
    if [ ! -d "$QWEN3VL_PATH" ]; then
        echo "ERROR: QWEN3VL_PATH not found: $QWEN3VL_PATH"
        echo "Set it via:  export QWEN3VL_PATH=/your/path/to/Qwen3-VL-32B-Instruct"
        exit 1
    fi
    echo "[serve] Qwen3-VL-32B-Instruct on port $QWEN3VL_PORT, GPU $CUDA_DEV"
    CUDA_VISIBLE_DEVICES=$CUDA_DEV vllm serve "$QWEN3VL_PATH" \
        --host 0.0.0.0 --port $QWEN3VL_PORT \
        --served-model-name Qwen/Qwen3-VL-32B-Instruct \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization $GPU_UTIL \
        --max-model-len $SHORT_CONTEXT \
        --dtype bfloat16
    ;;

  stop)
    echo "[serve] killing any local vllm serve processes"
    pkill -f "vllm serve" || true
    sleep 2
    ps -ef | grep "vllm serve" | grep -v grep || echo "  none running"
    ;;

  status)
    echo "=== vllm processes ==="
    ps -ef | grep "vllm serve" | grep -v grep || echo "  none running"
    echo ""
    echo "=== port $GEMMA_PORT (Gemma) ==="
    curl -sS http://localhost:$GEMMA_PORT/v1/models 2>/dev/null | head -c 300 || echo "  not responding"
    echo ""
    echo "=== port $QWEN3VL_PORT (Qwen3-VL) ==="
    curl -sS http://localhost:$QWEN3VL_PORT/v1/models 2>/dev/null | head -c 300 || echo "  not responding"
    ;;

  *)
    cat <<EOF
Usage: bash scripts/serve_teacher.sh <command>

Commands:
  gemma        Serve Gemma-4-31B-it on port 8003 (single GPU, 90 GB).
  gemma-tp4    Serve Gemma with tensor-parallel=4 (4× 24 GB cards).
  qwen3vl      Serve Qwen3-VL-32B-Instruct on port 8004 (single GPU).
  stop         Kill any running vllm serve process.
  status       Show running vllm + ping the two ports.

Set CUDA_VISIBLE_DEVICES, GEMMA_PATH, QWEN3VL_PATH as needed.

Examples:
  CUDA_VISIBLE_DEVICES=0  GEMMA_PATH=/models/gemma-4-31b-it    bash scripts/serve_teacher.sh gemma
  CUDA_VISIBLE_DEVICES=1  QWEN3VL_PATH=/models/Qwen3-VL-32B    bash scripts/serve_teacher.sh qwen3vl
EOF
    exit 1
    ;;
esac
