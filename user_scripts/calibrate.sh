#!/bin/bash

uv run lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5AA90179741 \
  --teleop.id=leader \
  --teleop.calibration_dir="./calibrate"

uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AA90180651 \
  --robot.id=follower \
  --robot.calibration_dir="./calibrate"
