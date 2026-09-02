#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=expert_ground_lora
ARGS=()
while (($#)); do
  case "$1" in
    --stage)
      case "${2:?--stage requires bootstrap or final}" in
        bootstrap) CONFIG=expert_ground_lora_bootstrap ;;
        final) CONFIG=expert_ground_lora ;;
        *) echo "--stage must be bootstrap or final" >&2; exit 2 ;;
      esac
      shift 2 ;;
    --config) CONFIG="${2:?--config requires a name}"; shift 2 ;;
    -h|--help)
      echo "Usage: bash training/train_grounding_expert.sh --stage bootstrap|final [runner options]"
      echo "Run bootstrap first; final output adapter is expert_ground_lora."
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
exec "${SCRIPT_DIR}/distillation/expert_lora/scripts/run_experiment.sh" "$CONFIG" "${ARGS[@]}"
