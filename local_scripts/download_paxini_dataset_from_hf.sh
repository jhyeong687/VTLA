#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HF_REPO_ID="${HF_REPO_ID:?Set HF_REPO_ID, e.g. jhyeong687/tissue}"
DATASET_ROOT="${DATASET_ROOT:-$PWD/outputs/${HF_REPO_ID##*/}}"

mkdir -p "$DATASET_ROOT"

.venv/bin/hf download "$HF_REPO_ID" \
  --repo-type dataset \
  --local-dir "$DATASET_ROOT" \
  --force-download

echo "Downloaded $HF_REPO_ID to $DATASET_ROOT"
