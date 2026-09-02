#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=grounding_exp5
ARGS=()
while (($#)); do
  case "$1" in
    --config) CONFIG="${2:?--config requires a name}"; shift 2 ;;
    -h|--help)
      echo "Usage: bash training/train_grounding_expert.sh --config NAME [runner options]"
      echo "Use grounding_bootstrap_942 first, then grounding_exp5 --init-lora PATH."
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec "${SCRIPT_DIR}/distillation/expert_lora/scripts/run_experiment.sh" "$CONFIG" "${ARGS[@]}"
