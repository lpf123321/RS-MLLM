#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:?usage: run_experiment.sh EXPERIMENT [runner options]}"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${PACKAGE_ROOT}/../../.." && pwd)"
CONFIG="${PACKAGE_ROOT}/configs/${EXPERIMENT}.json"
DATASET_ROOT="${RS_MLLM_TRAINING35_DATA:-${REPO_ROOT}/datasets/training35}"
SOURCE_ROOT="${RS_MLLM_DISTILLATION_CACHE:-${DATASET_ROOT}/.source}"
MODELS_ROOT="${RSMLLM_MODELS_ROOT:-${REPO_ROOT}/models}"

# README commands must work from a fresh login shell. Prefer an explicitly
# selected interpreter, otherwise keep an already usable environment or load
# the same conda environment as the five-stage launchers.
if [[ -n "${RS_MLLM_PYTHON:-}" ]]; then
  PYTHON_BIN="${RS_MLLM_PYTHON}"
elif python -c 'import torch, transformers' >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  # shellcheck source=../../../scripts/training/env.sh
  source "${REPO_ROOT}/scripts/training/env.sh"
  activate_conda
  PYTHON_BIN=python
fi

[[ -s "${CONFIG}" ]] || { echo "Unknown experiment: ${EXPERIMENT}" >&2; exit 2; }
if [[ ! -s "${SOURCE_ROOT}/ASSET_MANIFEST.json" && "${RS_MLLM_SKIP_DOWNLOAD:-0}" != "1" ]]; then
  mkdir -p "${SOURCE_ROOT}"
  command -v ms-hub >/dev/null 2>&1 || { echo "ms-hub is required to download training data" >&2; exit 2; }
  ms-hub download "${RS_MLLM_DISTILLATION_DATASET_ID:-Uchitachi/RS-MLLM-Distillation-Data}" \
    --repo-type dataset --local-dir "${SOURCE_ROOT}"
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/stage_training35_data.py" --expert-source "${SOURCE_ROOT}" --output "${DATASET_ROOT}"

export PYTHONPATH="${PACKAGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m expert_lora.runner \
  --config "${CONFIG}" --dataset-root "${DATASET_ROOT}" --models-root "${MODELS_ROOT}" "$@"
