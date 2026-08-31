#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:?usage: run_experiment.sh EXPERIMENT [runner options]}"
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ASSET_ROOT="${RS_MLLM_ASSET_ROOT:-${PACKAGE_ROOT}/.artifacts}"
PYTHON_BIN="${RS_MLLM_PYTHON:-python}"
CONFIG="${PACKAGE_ROOT}/configs/${EXPERIMENT}.json"

[[ -s "${CONFIG}" ]] || {
  echo "Unknown experiment: ${EXPERIMENT}" >&2
  echo "Available: general_exp3 grounding_bootstrap_942 grounding_exp1 grounding_exp4 grounding_exp5" >&2
  exit 2
}

# Public reproductions download from ModelScope automatically.  A developer with
# an already populated asset root can set RS_MLLM_SKIP_DOWNLOAD=1.
if [[ "${RS_MLLM_SKIP_DOWNLOAD:-0}" != "1" ]]; then
  RS_MLLM_ASSET_ROOT="${ASSET_ROOT}" "${SCRIPT_DIR}/download_assets.sh"
fi

export PYTHONPATH="${PACKAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m expert_lora.runner \
  --config "${CONFIG}" \
  --asset-root "${ASSET_ROOT}" \
  "$@"
