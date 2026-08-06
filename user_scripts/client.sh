#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-mps}"
FPS="${FPS:-10}"
TASK="${TASK:-Pick one cup from the stacked cups and place it on the left side.}"
POLICY_TYPE="${POLICY_TYPE:-pi05}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-${ACTION_PER_CHUNK:-50}}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"
DEBUG_VISUALIZE_QUEUE_SIZE="${DEBUG_VISUALIZE_QUEUE_SIZE:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/cu.usbmodem5AA90180651}"
TACTILE_PORT="${TACTILE_PORT:-/dev/cu.usbmodem750A687485311}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"
CALIBRATION_DIR="${CALIBRATION_DIR:-./calibrate}"

CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
CAMERA_FPS="${CAMERA_FPS:-30}"
SEND_IMAGE_WIDTH="${SEND_IMAGE_WIDTH:-224}"
SEND_IMAGE_HEIGHT="${SEND_IMAGE_HEIGHT:-224}"
BELLY_CAMERA_INDEX="${BELLY_CAMERA_INDEX:-2}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-3}"
TOP_CAMERA_INDEX="${TOP_CAMERA_INDEX:-1}"

TACTILE_BAUDRATE="${TACTILE_BAUDRATE:-921600}"
TACTILE_SCALE="${TACTILE_SCALE:-0.1}"
TACTILE_MODE="${TACTILE_MODE:-z}"
TACTILE_NUM_TAXELS="${TACTILE_NUM_TAXELS:-154}"
TACTILE_REPRESENTATION="${TACTILE_REPRESENTATION:-stacked_9x9}"
TACTILE_MOCK="${TACTILE_MOCK:-false}"

MODEL_REPO="${MODEL_REPO:-Leedohyeon/pi05-so101-paxini-cup-left-adapters}"
MODEL_ROOT="${MODEL_ROOT:-hf_models}"
MODEL_TYPE="${MODEL_TYPE:-gated}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

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

MODEL_REPO_NAME="${MODEL_REPO##*/}"
MODEL_REPO_DIR="${MODEL_REPO_DIR:-$MODEL_ROOT/$MODEL_REPO_NAME}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-$MODEL_REPO_DIR/$CHECKPOINT_DIR}"

if [[ -z "${ROBOT_CAMERAS:-}" ]]; then
  ROBOT_CAMERAS="{ base_0_rgb: {type: opencv, index_or_path: $BELLY_CAMERA_INDEX, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $CAMERA_FPS}, left_wrist_0_rgb: {type: opencv, index_or_path: $WRIST_CAMERA_INDEX, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $CAMERA_FPS}, right_wrist_0_rgb: {type: opencv, index_or_path: $TOP_CAMERA_INDEX, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $CAMERA_FPS}}"
fi

echo "Connecting to async policy server: $SERVER_ADDRESS"
echo "Model type: $MODEL_TYPE"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Pretrained path sent to server: $PRETRAINED_NAME_OR_PATH"
echo "Camera capture: ${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS} fps"
echo "Images sent to server: ${SEND_IMAGE_WIDTH}x${SEND_IMAGE_HEIGHT}"

exec uv run --extra async --extra core_scripts --extra paxini python -m lerobot.async_inference.robot_client \
  --server_address="$SERVER_ADDRESS" \
  --robot.type=paxini_so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.calibration_dir="$CALIBRATION_DIR" \
  --robot.cameras="$ROBOT_CAMERAS" \
  --robot.tactile_mock="$TACTILE_MOCK" \
  --robot.tactile_port="$TACTILE_PORT" \
  --robot.tactile_baudrate="$TACTILE_BAUDRATE" \
  --robot.tactile_taxel_scale="$TACTILE_SCALE" \
  --robot.tactile_taxel_value_mode="$TACTILE_MODE" \
  --robot.tactile_num_taxels="$TACTILE_NUM_TAXELS" \
  --robot.tactile_representation="$TACTILE_REPRESENTATION" \
  --task="$TASK" \
  --policy_type="$POLICY_TYPE" \
  --pretrained_name_or_path="$PRETRAINED_NAME_OR_PATH" \
  --policy_device="$POLICY_DEVICE" \
  --client_device="$CLIENT_DEVICE" \
  --actions_per_chunk="$ACTIONS_PER_CHUNK" \
  --chunk_size_threshold="$CHUNK_SIZE_THRESHOLD" \
  --fps="$FPS" \
  --image_resize_width="$SEND_IMAGE_WIDTH" \
  --image_resize_height="$SEND_IMAGE_HEIGHT" \
  --aggregate_fn_name="$AGGREGATE_FN_NAME" \
  --display_data="$DISPLAY_DATA" \
  --debug_visualize_queue_size="$DEBUG_VISUALIZE_QUEUE_SIZE"
