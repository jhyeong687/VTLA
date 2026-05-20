#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LEADER_PORT="${LEADER_PORT:-/dev/cu.usbmodem5AA90179741}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/cu.usbmodem5AA90180651}"
TACTILE_PORT="${TACTILE_PORT:-/dev/cu.usbmodem750A687485311}"
TELEOP_ID="${TELEOP_ID:-leader}"

DATASET_REPO_ID="${DATASET_REPO_ID:-local/paxini_tactile}"
DATASET_ROOT="${DATASET_ROOT:-$PWD/outputs/paxini_tactile_dataset}"
TASK="${TASK:-move the SO-101 arm while recording Paxini tactile response}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-300}"
RESET_TIME_S="${RESET_TIME_S:-5}"
FPS="${FPS:-30}"
TACTILE_BAUDRATE="${TACTILE_BAUDRATE:-921600}"
TACTILE_SCALE="${TACTILE_SCALE:-0.1}"
TACTILE_MODE="${TACTILE_MODE:-z}"
TACTILE_NUM_TAXELS="${TACTILE_NUM_TAXELS:-154}"
TACTILE_REPRESENTATION="${TACTILE_REPRESENTATION:-stacked_9x9}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"

rm -rf "$DATASET_ROOT"

exec .venv/bin/lerobot-record \
  --robot.discover_packages_path=lerobot_robot_paxini \
  --robot.type=paxini_so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower_arm \
  --robot.tactile_mock=false \
  --robot.tactile_port="$TACTILE_PORT" \
  --robot.tactile_baudrate="$TACTILE_BAUDRATE" \
  --robot.tactile_taxel_scale="$TACTILE_SCALE" \
  --robot.tactile_taxel_value_mode="$TACTILE_MODE" \
  --robot.tactile_num_taxels="$TACTILE_NUM_TAXELS" \
  --robot.tactile_representation="$TACTILE_REPRESENTATION" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$TELEOP_ID" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.fps="$FPS" \
  --dataset.push_to_hub=false \
  --dataset.video=false \
  --display_data="$DISPLAY_DATA"
