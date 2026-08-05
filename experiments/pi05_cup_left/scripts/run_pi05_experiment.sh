#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?Usage: $0 METHOD [steps]}"
STEPS="${2:-10}"

REPO_DIR="${REPO_DIR:-$(pwd)}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/vtla/data/paxini_cup_left_60}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/paxini_cup_left_60}"
RUN_ROOT="${RUN_ROOT:-/workspace/vtla/outputs/pi05_cup_left}"
PI05_BASE_PATH="${PI05_BASE_PATH:-/workspace/vtla/hf_models/pi05_base}"
RENAME_MAP='{"observation.images.belly":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb","observation.images.top":"observation.images.right_wrist_0_rgb"}'
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-false}"
LOG_FREQ="${LOG_FREQ:-1}"
SAVE_FREQ="${SAVE_FREQ:-$STEPS}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/.cache/lerobot}"
export TORCH_HOME="${TORCH_HOME:-/workspace/vtla/torch_cache}"
export WANDB_DIR="${WANDB_DIR:-/workspace/vtla/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "$METHOD" in
  A0)
    JOB_NAME="A0_no_tactile"
    TACTILE_MODE="none"
    ;;
  A15)
    JOB_NAME="A15_cnn_tactile_prefix"
    TACTILE_MODE="prefix"
    ;;
  A2)
    JOB_NAME="A2_gated_action_tactile"
    TACTILE_MODE="action_gated"
    ;;
  *)
    echo "Unknown METHOD '$METHOD'. Expected A0, A15, or A2." >&2
    exit 2
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$RUN_ROOT/${JOB_NAME}_${STEPS}steps_$STAMP"

cd "$REPO_DIR"

.venv/bin/lerobot-train \
  --policy.path="$PI05_BASE_PATH" \
  --policy.push_to_hub=false \
  --policy.repo_id="local/${JOB_NAME}" \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.tactile_injection_mode="$TACTILE_MODE" \
  --policy.tactile_encoder_type=cnn \
  --policy.tactile_n_tokens=1 \
  --policy.tactile_dropout=0.15 \
  --policy.tactile_contact_threshold=0.05 \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.use_imagenet_stats=true \
  --rename_map="$RENAME_MAP" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --prefetch_factor="$PREFETCH_FACTOR" \
  --persistent_workers="$PERSISTENT_WORKERS" \
  --eval_freq=0 \
  --log_freq="$LOG_FREQ" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --wandb.enable=false \
  --peft.method=LORA \
  --peft.r=8
