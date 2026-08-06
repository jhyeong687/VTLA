#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET_REPO_ID="${DATASET_REPO_ID:-local/paxini_cup_left_60}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/vtla/data/paxini_cup_left_60}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/vtla/hf_models}"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-$MODEL_ROOT/pi05-so101-paxini-cup-left-adapters}"
EPISODE="${EPISODE:-0}"
STRIDE="${STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_SEEDS="${NUM_SEEDS:-1}"
TEACHER_LOSS_REPEATS="${TEACHER_LOSS_REPEATS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/vtla/outputs/offline_eval/episode_${EPISODE}_$(date +%Y%m%d_%H%M%S)}"

A0_CHECKPOINT="all_checkpoints/pi05_cup_left/A0_no_tactile_4000steps_20260629_151550/checkpoints/020000/pretrained_model"
A15_CHECKPOINT="all_checkpoints/pi05_cup_left/A15_cnn_tactile_prefix_4000steps_20260629_163919/checkpoints/020000/pretrained_model"
A2_CHECKPOINT="all_checkpoints/pi05_cup_left/A2_gated_action_tactile_4000steps_20260629_180545/checkpoints/020000/pretrained_model"

PREPARE_ONLY=true MODEL_TYPE=none CHECKPOINT_DIR="$A0_CHECKPOINT" bash user_scripts/server.sh
PREPARE_ONLY=true MODEL_TYPE=cnn CHECKPOINT_DIR="$A15_CHECKPOINT" bash user_scripts/server.sh
PREPARE_ONLY=true MODEL_TYPE=gated CHECKPOINT_DIR="$A2_CHECKPOINT" bash user_scripts/server.sh

exec uv run --extra dataset --extra async --extra pi --extra peft lerobot-offline-action-eval \
  --dataset-repo-id="$DATASET_REPO_ID" \
  --dataset-root="$DATASET_ROOT" \
  --episode="$EPISODE" \
  --model="A0=$MODEL_REPO_DIR/$A0_CHECKPOINT" \
  --model="A15=$MODEL_REPO_DIR/$A15_CHECKPOINT" \
  --model="A2=$MODEL_REPO_DIR/$A2_CHECKPOINT" \
  --output-dir="$OUTPUT_DIR" \
  --device="$DEVICE" \
  --stride="$STRIDE" \
  --max-frames="$MAX_FRAMES" \
  --num-seeds="$NUM_SEEDS" \
  --teacher-loss-repeats="$TEACHER_LOSS_REPEATS"
