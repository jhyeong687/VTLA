#!/bin/bash

TOP_CAMERA_NAME="C270 HD WEBCAM"
BELLY_CAMERA_NAME="USB Camera"
WRIST_CAMERA_NAME="USB RGB Camera"

FOLLOWER_PORT="/dev/tty.usbmodem5AA90180651"
LEADER_PORT="/dev/tty.usbmodem5AA90179741"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

camera_json=$(get_avfoundation_device_map)
top_camera_index=$(echo "$camera_json" | find_camera_index_by_name "$TOP_CAMERA_NAME")
belly_camera_index=$(echo "$camera_json" | find_camera_index_by_name "$BELLY_CAMERA_NAME")
wrist_camera_index=$(echo "$camera_json" | find_camera_index_by_name "$WRIST_CAMERA_NAME")

CAMERA_JSON_STR="{
    top: {type: opencv, index_or_path: $top_camera_index, width: 1280, height: 720, fps: 30},
    wrist: {type: opencv, index_or_path: $wrist_camera_index, width: 1280, height: 720, fps: 30},
    belly: {type: opencv, index_or_path: $belly_camera_index, width: 1280, height: 720, fps: 30}
}"

if [[ -z "$top_camera_index" ]]; then
  echo "Error: Could not find camera index for '$TOP_CAMERA_NAME'"
  exit 1
fi
if [[ -z "$belly_camera_index" ]]; then
  echo "Error: Could not find camera index for '$BELLY_CAMERA_NAME'"
  exit 1
fi
if [[ -z "$wrist_camera_index" ]]; then
  echo "Error: Could not find camera index for '$WRIST_CAMERA_NAME'"
  exit 1
fi

uv run lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=leader \
  --display_data=true \
  --robot.calibration_dir="./calibrate" \
  --teleop.calibration_dir="./calibrate" \
  --robot.cameras="$CAMERA_JSON_STR"
