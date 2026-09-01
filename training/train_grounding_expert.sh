#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/distillation/expert_lora/scripts/run_experiment.sh"

usage() {
  cat <<'EOF'
Usage: bash training/train_grounding_expert.sh [runner options]

Continue the Grounding Expert with the retained Grounding Exp5 recipe:
  washed VRSBench + VRSBench-new + XLRS official/new,
  43,838 mixed grounding records in total.

The launcher downloads missing ModelScope assets, composes
Qwen3.5-4B + grounding_expert_delta, loads the retained 942-example
bootstrap LoRA, and continues training a new LoRA adapter.

Common options:
  --gpus 0,1,2,3,4,5,6,7  Visible GPUs (the retained recipe requires 8)
  --output-root PATH       Runtime-model and adapter output directory
  --prepare-only           Validate data and compose the runtime model only
  --dry-run                Validate inputs and print commands without training
  --max-updates N          Stop after N optimizer updates (smoke testing only)

Environment:
  RS_MLLM_ASSET_ROOT        Override the download/asset directory
  RS_MLLM_SKIP_DOWNLOAD=1   Use assets already present locally
  RS_MLLM_PYTHON            Python executable (default: python)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -x "${LAUNCHER}" ]] || {
  echo "Expert LoRA launcher is missing or not executable: ${LAUNCHER}" >&2
  exit 2
}

exec "${LAUNCHER}" grounding_exp5 "$@"
