#!/usr/bin/env bash
# Launch 3 parallel Gemini Pro 2.5 annotation passes (one per variant).
# Each pass uses a DIFFERENT GCP project (its own quota) via a
# service-account JSON key from ./keys/key-{1,2,3}.json.
#
# Prerequisites:
#   - Run _setup_3_gcp_projects.sh first  (creates the 3 projects + keys)
#   - Python env: conda activate navlm_v2 (or use the python referenced below)
#   - pip install google-auth google-auth-oauthlib  (if not present)
#
# Run:
#   bash _launch_3_annotation_passes.sh
#
# Logs:
#   logs/annot_given.log
#   logs/annot_derived.log
#   logs/annot_implicit.log
#
# Watch:
#   tail -f logs/annot_*.log
#
# Cost / time:
#   ~$30 per pass × 3 = ~$90
#   ~11 h wall (parallel, vs ~33 h sequential)

set -uo pipefail

PY="C:/Users/z0502/anaconda3/envs/navlm_v2/python.exe"
KEY_DIR="./keys"
LOG_DIR="./logs"
PROJECT_PREFIX="navlm-annot"

mkdir -p "$LOG_DIR"

# Map: variant → (key#, project#)
declare -A KEY_NUM=( [given]=1 [derived]=2 [implicit]=3 )

PIDS=()

for variant in given derived implicit; do
  i=${KEY_NUM[$variant]}
  KEY_FILE="${KEY_DIR}/key-${i}.json"
  PROJECT="${PROJECT_PREFIX}-${i}-26"
  LOG_FILE="${LOG_DIR}/annot_${variant}.log"

  if [[ ! -f "$KEY_FILE" ]]; then
    echo "ERROR: key file not found: $KEY_FILE"
    echo "  Run _setup_3_gcp_projects.sh first."
    exit 1
  fi

  echo "── launching ${variant}  (project=${PROJECT}, key=${KEY_FILE})"

  # Each background process inherits an independent env so it uses its
  # own credentials + project. PYTHONUTF8/IOENCODING for Windows console.
  (
    export PYTHONUTF8=1
    export PYTHONIOENCODING=utf-8
    export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE"
    export GCP_PROJECT="$PROJECT"
    "$PY" -u -m src.a2_annotate \
        --variant "$variant" \
        --limit 0 \
        --max-tokens 4096 \
        --resume \
        > "$LOG_FILE" 2>&1
  ) &
  PIDS+=($!)
  echo "  pid=$! → ${LOG_FILE}"
done

echo ""
echo "All 3 passes launched.  Watch progress:"
echo "  tail -f ${LOG_DIR}/annot_*.log"
echo ""
echo "Wait for all to complete:"
echo "  wait ${PIDS[*]}"

# Block until all finish (will keep the shell alive)
wait "${PIDS[@]}"
echo ""
echo "All 3 annotation passes finished."
