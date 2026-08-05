#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
PI05_BASE_PATH="${PI05_BASE_PATH:-/workspace/vtla/hf_models/pi05_base}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/workspace/vtla/hf_models/paligemma-3b-pt-224-tokenizer}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/.cache/lerobot}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO_DIR"

.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

base = Path(os.environ.get("PI05_BASE_PATH", "/workspace/vtla/hf_models/pi05_base"))
tokenizer_dir = Path(os.environ.get("TOKENIZER_DIR", "/workspace/vtla/hf_models/paligemma-3b-pt-224-tokenizer"))
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

tokenizer_dir.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", token=token)
tokenizer.save_pretrained(tokenizer_dir)

preprocessor_path = base / "policy_preprocessor.json"
with preprocessor_path.open("r", encoding="utf-8") as f:
    preprocessor = json.load(f)

for step in preprocessor["steps"]:
    if step.get("registry_name") == "tokenizer_processor":
        step["config"]["tokenizer_name"] = str(tokenizer_dir)

with preprocessor_path.open("w", encoding="utf-8") as f:
    json.dump(preprocessor, f, indent=2)
    f.write("\n")

print(f"saved tokenizer to {tokenizer_dir}")
print(f"patched {preprocessor_path}")
PY
