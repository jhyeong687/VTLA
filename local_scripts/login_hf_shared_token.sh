#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN before running this script."
  echo "Example:"
  echo "  export HF_TOKEN='paste-token-here'"
  echo "  bash local_scripts/login_hf_shared_token.sh"
  exit 1
fi

.venv/bin/hf auth login --token "$HF_TOKEN" --add-to-git-credential
.venv/bin/hf auth whoami
