#!/usr/bin/env bash
# One-shot setup for 3 parallel Vertex AI annotation passes.
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. `gcloud auth login`  (use the Google account that owns the
#                            billing account you'll attach)
#   3. Get your billing account ID:
#        gcloud beta billing accounts list
#      Looks like:  01ABCD-EF1234-567890
#   4. Edit BILLING_ID below before running.
#
# Run:
#   bash _setup_3_gcp_projects.sh
#
# What it does (per project, 3 times):
#   - Creates a new GCP project (id: navlm-annot-{1,2,3}-26)
#   - Links it to your billing account
#   - Enables the Vertex AI API
#   - Creates a service account `runner@PROJECT.iam.gserviceaccount.com`
#   - Grants it `roles/aiplatform.user`
#   - Creates a JSON key file under ./keys/key-{1,2,3}.json
#
# After it finishes, run the 3 parallel annotation passes
# (see commands printed at the end).

set -euo pipefail

#────────────────────────────────────────────────────────────────────
# EDIT THIS BEFORE RUNNING
#────────────────────────────────────────────────────────────────────
BILLING_ID="01DF8D-AAE976-C395FE"          # e.g. "01ABCD-EF1234-567890"
PROJECT_PREFIX="navlm-annot"
SA_NAME="runner"
KEY_DIR="./keys"

if [[ -z "$BILLING_ID" ]]; then
  echo "ERROR: set BILLING_ID at the top of this script first."
  echo "  Find it with:  gcloud beta billing accounts list"
  exit 1
fi

mkdir -p "$KEY_DIR"

for i in 1 2 3; do
  PROJ="${PROJECT_PREFIX}-${i}-26"
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "Project ${i}/3: ${PROJ}"
  echo "════════════════════════════════════════════════════════════════"

  # 1. create project
  if gcloud projects describe "$PROJ" >/dev/null 2>&1; then
    echo "  ✓ project ${PROJ} already exists (skipping create)"
  else
    gcloud projects create "$PROJ" --name="NavLM Annot $i"
  fi

  # 2. link billing
  gcloud beta billing projects link "$PROJ" \
      --billing-account="$BILLING_ID"

  # 3. enable Vertex AI API
  gcloud services enable aiplatform.googleapis.com --project="$PROJ"

  # 4. create service account
  SA_EMAIL="${SA_NAME}@${PROJ}.iam.gserviceaccount.com"
  if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJ" \
       >/dev/null 2>&1; then
    echo "  ✓ service account ${SA_NAME} already exists"
  else
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="NavLM annotation runner $i" \
        --project="$PROJ"
  fi

  # 5. grant Vertex AI user role
  gcloud projects add-iam-policy-binding "$PROJ" \
      --member="serviceAccount:${SA_EMAIL}" \
      --role="roles/aiplatform.user" \
      --condition=None >/dev/null

  # 6. create + download JSON key
  KEY_FILE="${KEY_DIR}/key-${i}.json"
  if [[ -f "$KEY_FILE" ]]; then
    echo "  ✓ key file already exists: ${KEY_FILE}  (skipping re-create)"
  else
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SA_EMAIL"
    chmod 600 "$KEY_FILE"
    echo "  ✓ key written → ${KEY_FILE}"
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "DONE.  3 projects ready.  Launch parallel annotation with:"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  bash _launch_3_annotation_passes.sh"
