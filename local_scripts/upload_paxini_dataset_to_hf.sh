#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HF_REPO_ID="${HF_REPO_ID:?Set HF_REPO_ID, e.g. jhyeong687/tissue}"
DATASET_ROOT="${DATASET_ROOT:-$PWD/outputs/${HF_REPO_ID##*/}}"
DATASET_PRIVATE="${DATASET_PRIVATE:-true}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload Paxini dataset update}"

if [[ ! -d "$DATASET_ROOT/meta" || ! -d "$DATASET_ROOT/data" ]]; then
  echo "Dataset root does not look like a LeRobot dataset: $DATASET_ROOT"
  exit 1
fi

visibility_flag="--private"
if [[ "$DATASET_PRIVATE" != "true" ]]; then
  visibility_flag="--public"
fi

.venv/bin/hf repos create "$HF_REPO_ID" --repo-type dataset "$visibility_flag" --exist-ok
.venv/bin/hf upload "$HF_REPO_ID" "$DATASET_ROOT" . \
  --repo-type dataset \
  "$visibility_flag" \
  --commit-message "$COMMIT_MESSAGE"

echo "Uploaded $DATASET_ROOT to https://huggingface.co/datasets/$HF_REPO_ID"
