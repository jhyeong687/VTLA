#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HF_NAMESPACE="${HF_NAMESPACE:-}"
DATASET_PRIVATE="${DATASET_PRIVATE:-true}"

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

visibility_flag="--private"
if [[ "$DATASET_PRIVATE" != "true" ]]; then
  visibility_flag="--public"
fi

for dataset_name in plastic tissue sponge; do
  repo_id="$HF_NAMESPACE/$dataset_name"
  echo "Creating or checking dataset repo: $repo_id"
  .venv/bin/hf repos create "$repo_id" --repo-type dataset "$visibility_flag" --exist-ok
done

echo
echo "Dataset repos are ready:"
echo "  $HF_NAMESPACE/plastic"
echo "  $HF_NAMESPACE/tissue"
echo "  $HF_NAMESPACE/sponge"
