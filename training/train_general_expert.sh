#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=general_exp7
ARGS=()
while (($#)); do
  case "$1" in
    --config) CONFIG="${2:?--config requires a name}"; shift 2 ;;
    -h|--help)
      echo "Usage: bash training/train_general_expert.sh [--config general_exp7] [runner options]"
      echo "Default: exact 15,480-record General Exp7; --gpus is a GPU count."
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec "${SCRIPT_DIR}/distillation/expert_lora/scripts/run_experiment.sh" "$CONFIG" "${ARGS[@]}"
