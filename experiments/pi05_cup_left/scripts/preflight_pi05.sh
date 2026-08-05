#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
PI05_BASE_PATH="${PI05_BASE_PATH:-/workspace/vtla/hf_models/pi05_base}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/.cache/lerobot}"
export TORCH_HOME="${TORCH_HOME:-/workspace/vtla/torch_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO_DIR"

test -f "$PI05_BASE_PATH/model.safetensors"
test -f "$PI05_BASE_PATH/config.json"
test -f "$PI05_BASE_PATH/policy_preprocessor.json"
test -f "$PI05_BASE_PATH/policy_postprocessor.json"

.venv/bin/python - <<'PY'
import os

from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.policies.factory import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

base = os.environ.get("PI05_BASE_PATH", "/workspace/vtla/hf_models/pi05_base")
pre = PolicyProcessorPipeline.from_pretrained(
    base,
    config_filename="policy_preprocessor.json",
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)
post = PolicyProcessorPipeline.from_pretrained(
    base,
    config_filename="policy_postprocessor.json",
    to_transition=policy_action_to_transition,
    to_output=transition_to_policy_action,
)
print("pre_steps", [type(step).__name__ for step in pre.steps])
print("post_steps", [type(step).__name__ for step in post.steps])
PY
