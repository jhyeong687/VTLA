#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BATCH_SIZE="${BATCH_SIZE:-2}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
LOG_FREQ="${LOG_FREQ:-1}" \
SAVE_FREQ="${SAVE_FREQ:-$STEPS}" \
bash "$SCRIPT_DIR/queue_main.sh" "$STEPS"
