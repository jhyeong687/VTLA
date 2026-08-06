#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.0333333333}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"

MODEL_REPO="${MODEL_REPO:-Leedohyeon/pi05-so101-paxini-cup-left-adapters}"
BASE_REPO="${BASE_REPO:-lerobot/pi05_base}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/vtla/hf_models}"
PI05_BASE_PATH="${PI05_BASE_PATH:-$MODEL_ROOT/pi05_base}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$MODEL_ROOT/paligemma-3b-pt-224-tokenizer}"
MODEL_TYPE="${MODEL_TYPE:-gated}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
PREPARE_ONLY="${PREPARE_ONLY:-false}"

case "$MODEL_TYPE" in
  none | cnn | gated)
    ;;
  *)
    echo "Unknown MODEL_TYPE='$MODEL_TYPE'. Expected one of: none, cnn, gated." >&2
    exit 2
    ;;
esac

if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "CHECKPOINT_DIR is required." >&2
  echo "Example: CHECKPOINT_DIR=all_checkpoints/pi05_cup_left/<run>/checkpoints/<step>/pretrained_model" >&2
  exit 2
fi

if [[ "$CHECKPOINT_DIR" == /* || "$CHECKPOINT_DIR" == *".."* ]]; then
  echo "CHECKPOINT_DIR must be a relative path inside $MODEL_REPO; got '$CHECKPOINT_DIR'." >&2
  exit 2
fi

MODEL_REPO_DIR="${MODEL_REPO_DIR:-$MODEL_ROOT/pi05-so101-paxini-cup-left-adapters}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-$MODEL_REPO_DIR/$CHECKPOINT_DIR}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/.cache/lerobot}"
export TORCH_HOME="${TORCH_HOME:-/workspace/vtla/torch_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PI05_BASE_PATH
export TOKENIZER_DIR
export MODEL_ROOT
export MODEL_REPO
export BASE_REPO
export MODEL_REPO_DIR
export MODEL_TYPE
export CHECKPOINT_DIR
export PRETRAINED_NAME_OR_PATH

uv run --extra async --extra pi --extra peft python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

base_repo = os.environ["BASE_REPO"]
model_repo = os.environ["MODEL_REPO"]
model_type = os.environ["MODEL_TYPE"]
base_path = Path(os.environ["PI05_BASE_PATH"])
tokenizer_dir = Path(os.environ["TOKENIZER_DIR"])
model_repo_dir = Path(os.environ["MODEL_REPO_DIR"])
checkpoint_dir = os.environ["CHECKPOINT_DIR"]
adapter_path = Path(os.environ["PRETRAINED_NAME_OR_PATH"])
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

if not (base_path / "model.safetensors").is_file():
    base_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=base_repo,
        local_dir=base_path,
        allow_patterns=["*.json", "*.safetensors", "README.md"],
        token=token,
    )

if not adapter_path.is_dir():
    model_repo_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_repo,
        local_dir=model_repo_dir,
        allow_patterns=[f"{checkpoint_dir}/*"],
        token=token,
    )

if not (tokenizer_dir / "tokenizer_config.json").is_file():
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", token=token)
    tokenizer.save_pretrained(tokenizer_dir)

adapter_config_path = adapter_path / "adapter_config.json"
if adapter_config_path.is_file():
    adapter_config = json.loads(adapter_config_path.read_text())
    adapter_config["base_model_name_or_path"] = str(base_path)
    adapter_config_path.write_text(json.dumps(adapter_config, indent=2) + "\n")

config_path = adapter_path / "config.json"
if config_path.is_file():
    config = json.loads(config_path.read_text())
    config["pretrained_path"] = str(base_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
else:
    config = {}

preprocessor_path = adapter_path / "policy_preprocessor.json"
if preprocessor_path.is_file():
    preprocessor = json.loads(preprocessor_path.read_text())
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            step.setdefault("config", {})["tokenizer_name"] = str(tokenizer_dir)
    preprocessor_path.write_text(json.dumps(preprocessor, indent=2) + "\n")

required = [
    base_path / "model.safetensors",
    adapter_path / "adapter_config.json",
    adapter_path / "adapter_model.safetensors",
    adapter_path / "config.json",
    adapter_path / "policy_preprocessor.json",
    adapter_path / "policy_postprocessor.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Missing required model files:\n" + "\n".join(missing))

actual_injection_mode = config.get("tactile_injection_mode")
actual_encoder_type = config.get("tactile_encoder_type")
expected_by_model_type = {
    "none": {"tactile_injection_mode": "none"},
    "cnn": {"tactile_injection_mode": "prefix", "tactile_encoder_type": "cnn"},
    "gated": {"tactile_injection_mode": "action_gated"},
}
expected = expected_by_model_type[model_type]
errors = []
if actual_injection_mode != expected["tactile_injection_mode"]:
    errors.append(
        "tactile_injection_mode="
        f"{actual_injection_mode!r}, expected {expected['tactile_injection_mode']!r}"
    )
expected_encoder_type = expected.get("tactile_encoder_type")
if expected_encoder_type is not None and actual_encoder_type != expected_encoder_type:
    errors.append(f"tactile_encoder_type={actual_encoder_type!r}, expected {expected_encoder_type!r}")
if errors:
    raise ValueError(f"Checkpoint {adapter_path} does not match MODEL_TYPE={model_type!r}: " + "; ".join(errors))

print(f"Prepared PI05 base: {base_path}")
print(f"Prepared adapter: {adapter_path}")
print(f"Validated model type: {model_type}")
PY

if [[ "$PREPARE_ONLY" == "true" ]]; then
  echo "Checkpoint preparation complete; PREPARE_ONLY=true, so the policy server will not start."
  exit 0
fi

echo "Policy server will load adapter path sent by the client:"
echo "  $PRETRAINED_NAME_OR_PATH"

exec uv run --extra async --extra pi --extra peft python -m lerobot.async_inference.policy_server \
  --host="$HOST" \
  --port="$PORT" \
  --fps="$FPS" \
  --inference_latency="$INFERENCE_LATENCY" \
  --obs_queue_timeout="$OBS_QUEUE_TIMEOUT"
