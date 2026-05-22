#!/usr/bin/env bash
# Orchestrator — run the full NavLM toolbox for a city.
#
# Usage:
#   toolbox/run_city.sh <city>                       # uses videos already in data/cities/<city>/videos/
#   toolbox/run_city.sh <city> --local-dir PATH      # symlinks mp4s from PATH
#   toolbox/run_city.sh <city> --static-image-dir P  # skips video/frame step (MVP)
#
# Environment overrides:
#   FPS=0.1   FRAMES/sec (default 0.1 = one every 10s)
#   MAX_IMAGES=100
#   DEVICE=2  CUDA device index
#   EPOCHS=1

set -e
cd "$(dirname "$0")/.."

CITY="${1:-}"; shift || true
if [[ -z "$CITY" ]]; then
  echo "usage: $0 <city> [--local-dir PATH | --static-image-dir PATH | --url-list FILE]"
  exit 1
fi

FPS="${FPS:-0.1}"
MAX_IMAGES="${MAX_IMAGES:-100}"
DEVICE="${DEVICE:-2}"
EPOCHS="${EPOCHS:-1}"
DEDUP="${DEDUP:-0}"                   # set =1 for scene-adaptive sampling
PHASH_THRESHOLD="${PHASH_THRESHOLD:-10}"

EXTRA_FETCH=""
EXTRA_EXTRACT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-dir)          EXTRA_FETCH+=" --local-dir $2"; shift 2;;
    --url-list)           EXTRA_FETCH+=" --url-list $2"; shift 2;;
    --static-image-dir)   EXTRA_EXTRACT+=" --static-image-dir $2"; shift 2;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

echo "=== [1/4] fetch videos ==="
if [[ -n "$EXTRA_EXTRACT" ]]; then
  echo "(skipping: --static-image-dir mode)"
else
  python3 toolbox/fetch_videos.py --city "$CITY" $EXTRA_FETCH || {
    echo "⚠ no videos fetched; expect to use --static-image-dir"
    exit 2
  }
fi

echo "=== [2/4] extract frames ==="
EXTRACT_ARGS="--city $CITY --max-images $MAX_IMAGES $EXTRA_EXTRACT"
if [[ "$DEDUP" == "1" ]]; then
  EXTRACT_ARGS="$EXTRACT_ARGS --dedup --phash-threshold $PHASH_THRESHOLD"
  echo "  (scene-adaptive mode, phash_threshold=$PHASH_THRESHOLD)"
else
  EXTRACT_ARGS="$EXTRACT_ARGS --fps $FPS"
fi
python3 toolbox/extract_frames.py $EXTRACT_ARGS

echo "=== [3a/4] auto-annotate ==="
CUDA_VISIBLE_DEVICES="$DEVICE" python3 toolbox/auto_annotate.py --city "$CITY" --device "cuda:0"

echo "=== [3b/4] build SFT jsonl ==="
python3 toolbox/build_sft.py --city "$CITY"

echo "=== [4/4] train LoRA ==="
python3 toolbox/train_city_lora.py --city "$CITY" --epochs "$EPOCHS" --device "$DEVICE"

echo ""
echo "=== DONE ==="
echo "Adapter: $(pwd)/results/lora_$CITY/"
echo "Try it with:  python3 scripts/baseline_demo.py --image <any_img> --text 'Where am I?' "
echo "              (add PeftModel.from_pretrained(model, 'results/lora_$CITY') to the script)"
