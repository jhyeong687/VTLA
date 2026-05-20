#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LEADER_PORT="${LEADER_PORT:-/dev/cu.usbmodem5AA90179741}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/cu.usbmodem5AA90180651}"
FPS="${FPS:-60}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"

ARGS=(
  --robot.type=so101_follower
  --robot.port="$FOLLOWER_PORT"
  --robot.id=follower
  --teleop.type=so101_leader
  --teleop.port="$LEADER_PORT"
  --teleop.id=leader
  --fps="$FPS"
  --display_data="$DISPLAY_DATA"
)

if [[ "${TELEOP_TIME_S:-}" != "" ]]; then
  ARGS+=(--teleop_time_s="$TELEOP_TIME_S")
fi

exec .venv/bin/lerobot-teleoperate "${ARGS[@]}"
