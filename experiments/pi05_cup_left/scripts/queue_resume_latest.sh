#!/usr/bin/env bash
set -euo pipefail

TARGET_STEPS="${1:-8000}"
RUN_ROOT="${RUN_ROOT:-/workspace/vtla/outputs/pi05_cup_left}"
REPO_DIR="${REPO_DIR:-$(pwd)}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/.cache/lerobot}"
export TORCH_HOME="${TORCH_HOME:-/workspace/vtla/torch_cache}"
export WANDB_DIR="${WANDB_DIR:-/workspace/vtla/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO_DIR"

run_resume() {
  local label="$1"
  local pattern="$2"
  local run_dir
  run_dir="$(find "$RUN_ROOT" -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "$run_dir" ]]; then
    echo "No run directory found for $label with pattern $pattern" >&2
    return 1
  fi

  local latest_checkpoint
  latest_checkpoint="$(find "$run_dir/checkpoints" -maxdepth 1 -type d -regex '.*/[0-9][0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | sort -n | tail -1)"
  if [[ -z "$latest_checkpoint" ]]; then
    echo "No checkpoints found for $label in $run_dir/checkpoints" >&2
    return 1
  fi

  local config_path="$run_dir/checkpoints/$latest_checkpoint/pretrained_model/train_config.json"
  if [[ ! -f "$config_path" ]]; then
    echo "Missing checkpoint config for $label: $config_path" >&2
    return 1
  fi

  echo "=== Resuming $label to $TARGET_STEPS steps ==="
  echo "run_dir: $run_dir"
  echo "from_checkpoint: $latest_checkpoint"
  .venv/bin/lerobot-train \
    --config_path="$config_path" \
    --resume=true \
    --steps="$TARGET_STEPS" \
    --save_freq=1000 \
    --log_freq=25
}

run_resume A0 'A0_no_tactile_4000steps_*'
run_resume A1.5 'A1.5_cnn_tactile_prefix_4000steps_*'
run_resume A2 'A2_gated_action_tactile_4000steps_*'
