#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LEADER_PORT="${LEADER_PORT:-/dev/cu.usbmodem5AA90179741}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/cu.usbmodem5AA90180651}"
TACTILE_PORT="${TACTILE_PORT:-/dev/cu.usbmodem750A687485311}"
TELEOP_ID="${TELEOP_ID:-leader}"

DATASET_REPO_ID="${DATASET_REPO_ID:-local/paxini_tactile}"
DATASET_ROOT="${DATASET_ROOT:-$PWD/outputs/paxini_tactile_dataset}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DATASET_PRIVATE="${DATASET_PRIVATE:-true}"
STAMP_REPO_ID_ON_CREATE="${STAMP_REPO_ID_ON_CREATE:-false}"
TASK="${TASK:-move the SO-101 arm while recording Paxini tactile response}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-300}"
RESET_TIME_S="${RESET_TIME_S:-5}"
RESUME="${RESUME:-false}"
FPS="${FPS:-30}"
DATASET_VIDEO="${DATASET_VIDEO:-true}"
CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
BELLY_CAMERA_INDEX="${BELLY_CAMERA_INDEX:-2}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-3}"
TOP_CAMERA_INDEX="${TOP_CAMERA_INDEX:-1}"
if [[ -z "${ROBOT_CAMERAS:-}" ]]; then
  ROBOT_CAMERAS="{ belly: {type: opencv, index_or_path: $BELLY_CAMERA_INDEX, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $FPS}, wrist: {type: opencv, index_or_path: $WRIST_CAMERA_INDEX, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $FPS}, top: {type: opencv, index_or_path: $TOP_CAMERA_INDEX, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $FPS}}"
fi
TACTILE_BAUDRATE="${TACTILE_BAUDRATE:-921600}"
TACTILE_SCALE="${TACTILE_SCALE:-0.1}"
TACTILE_MODE="${TACTILE_MODE:-z}"
TACTILE_NUM_TAXELS="${TACTILE_NUM_TAXELS:-154}"
TACTILE_REPRESENTATION="${TACTILE_REPRESENTATION:-stacked_9x9}"
EXPORT_TACTILE_VIDEO="${EXPORT_TACTILE_VIDEO:-true}"
TACTILE_VIDEO_NORMALIZATION="${TACTILE_VIDEO_NORMALIZATION:-global}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"

if [[ "$RESUME" != "true" ]]; then
  rm -rf "$DATASET_ROOT"
fi

.venv/bin/lerobot-record \
  --robot.discover_packages_path=lerobot_robot_paxini \
  --robot.type=paxini_so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower_arm \
  --robot.cameras="$ROBOT_CAMERAS" \
  --robot.tactile_mock=false \
  --robot.tactile_port="$TACTILE_PORT" \
  --robot.tactile_baudrate="$TACTILE_BAUDRATE" \
  --robot.tactile_taxel_scale="$TACTILE_SCALE" \
  --robot.tactile_taxel_value_mode="$TACTILE_MODE" \
  --robot.tactile_num_taxels="$TACTILE_NUM_TAXELS" \
  --robot.tactile_representation="$TACTILE_REPRESENTATION" \
  --robot.calibration_dir="./calibrate" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$TELEOP_ID" \
  --teleop.calibration_dir="./calibrate" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.fps="$FPS" \
  --dataset.push_to_hub="$PUSH_TO_HUB" \
  --dataset.private="$DATASET_PRIVATE" \
  --dataset.stamp_repo_id_on_create="$STAMP_REPO_ID_ON_CREATE" \
  --dataset.video="$DATASET_VIDEO" \
  --resume="$RESUME" \
  --display_data="$DISPLAY_DATA"

if [[ "$EXPORT_TACTILE_VIDEO" == "true" ]]; then
  while IFS= read -r parquet_path; do
    rel_path="${parquet_path#"$DATASET_ROOT/data/"}"
    rel_no_ext="${rel_path%.parquet}"
    video_path="$DATASET_ROOT/tactile_viz/$rel_no_ext.mp4"
    .venv/bin/python local_scripts/export_paxini_tactile_sequence.py \
      --parquet "$parquet_path" \
      --output-dir "$DATASET_ROOT/tactile_viz/frames/$rel_no_ext" \
      --video "$video_path" \
      --fps "$FPS" \
      --normalization "$TACTILE_VIDEO_NORMALIZATION" \
      --no-frames \
      --overwrite
  done < <(find "$DATASET_ROOT/data" -name '*.parquet' -type f | sort)
fi
