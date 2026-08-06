#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-4000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_pi05_experiment.sh" A0 "$STEPS"
bash "$SCRIPT_DIR/run_pi05_experiment.sh" A1.5 "$STEPS"
bash "$SCRIPT_DIR/run_pi05_experiment.sh" A2 "$STEPS"
