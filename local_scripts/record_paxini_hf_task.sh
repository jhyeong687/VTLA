#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PAXINI_TASK="${PAXINI_TASK:-tissue}"
HF_NAMESPACE="${HF_NAMESPACE:-}"
DATASET_PRIVATE="${DATASET_PRIVATE:-true}"
SYNC_FROM_HUB="${SYNC_FROM_HUB:-true}"
UPLOAD_TO_HUB="${UPLOAD_TO_HUB:-true}"

if [[ -z "$HF_NAMESPACE" ]]; then
  HF_NAMESPACE="$(.venv/bin/hf auth whoami | sed -n 's/^user=//p' | head -1)"
  echo "HF_NAMESPACE is not set; defaulting to the logged-in user namespace: $HF_NAMESPACE"
  echo "Set HF_NAMESPACE explicitly if the dataset repos are under a different account."
fi

if [[ -z "$HF_NAMESPACE" ]]; then
  echo "Could not determine Hugging Face namespace."
  echo "Run: .venv/bin/hf auth login"
  echo "Or set: HF_NAMESPACE=<username-or-org>"
  exit 1
fi

case "$PAXINI_TASK" in
  plastic)
    dataset_name="plastic"
    task_text="Pick up the plastic object and place it in the box"
    ;;
  tissue)
    dataset_name="tissue"
    task_text="Pick up the tissue and place it nearby"
    ;;
  sponge)
    dataset_name="sponge"
    task_text="Pick up the makeup puff and place it in the box"
    ;;
  *)
    echo "Unknown PAXINI_TASK: $PAXINI_TASK"
    echo "Choose one of: plastic, tissue, sponge"
    exit 1
    ;;
esac

HF_REPO_ID="$HF_NAMESPACE/$dataset_name"
DATASET_ROOT="${DATASET_ROOT:-$PWD/outputs/$dataset_name}"
RESUME="${RESUME:-false}"

visibility_flag="--private"
if [[ "$DATASET_PRIVATE" != "true" ]]; then
  visibility_flag="--public"
fi

.venv/bin/hf repos create "$HF_REPO_ID" --repo-type dataset "$visibility_flag" --exist-ok
echo "Using Hugging Face dataset repo: $HF_REPO_ID"
echo "Using local dataset root: $DATASET_ROOT"

if [[ "$RESUME" == "true" && "$SYNC_FROM_HUB" == "true" ]]; then
  mkdir -p "$DATASET_ROOT"
  .venv/bin/hf download "$HF_REPO_ID" \
    --repo-type dataset \
    --local-dir "$DATASET_ROOT" \
    --force-download
  if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
    echo "No existing LeRobot dataset found on Hub for $HF_REPO_ID; starting a new dataset."
    RESUME=false
  fi
fi

DATASET_REPO_ID="$HF_REPO_ID" \
DATASET_ROOT="$DATASET_ROOT" \
TASK="$task_text" \
RESUME="$RESUME" \
PUSH_TO_HUB=false \
DATASET_PRIVATE="$DATASET_PRIVATE" \
STAMP_REPO_ID_ON_CREATE=false \
bash local_scripts/record_paxini_tactile.sh

if [[ "$UPLOAD_TO_HUB" == "true" ]]; then
  HF_REPO_ID="$HF_REPO_ID" \
  DATASET_ROOT="$DATASET_ROOT" \
  DATASET_PRIVATE="$DATASET_PRIVATE" \
  COMMIT_MESSAGE="Upload $dataset_name Paxini dataset update" \
  bash local_scripts/upload_paxini_dataset_to_hf.sh
fi
