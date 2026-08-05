#!/bin/bash

get_avfoundation_device_map() {
  ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | \
      grep 'AVFoundation indev @ ' | \
      python3 <( \
          cat <<'PYTHON'
import json
device_map = {}

while True:
    line = input()
    if "AVFoundation video devices:" in line:
        break

while True:
    line = input()
    if "AVFoundation audio devices:" in line:
        break
    if "AVFoundation indev @ " in line:
        device_id = line.split(' ')[4][1:-1]
        device_name = line.split('] ')[2]
        device_map[device_name] = device_id

print(json.dumps(device_map, indent=4))
PYTHON
      )
}

find_camera_index_by_name() {
  local target_name="$1"
  jq -r --arg name "$target_name" '.[$name] // empty'
}
