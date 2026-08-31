#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ASSET_ROOT="${RS_MLLM_ASSET_ROOT:-${PACKAGE_ROOT}/.artifacts}"

BASE_ID="${RS_MLLM_QWEN35_MODEL_ID:-Qwen/Qwen3.5-4B}"
DATA_ID="${RS_MLLM_DISTILLATION_DATASET_ID:-Uchitachi/RS-MLLM-Distillation-Data}"
CHECKPOINT_ID="${RS_MLLM_DISTILLATION_CHECKPOINT_ID:-Uchitachi/RS-MLLM-Distillation-Checkpoints}"

command -v ms-hub >/dev/null 2>&1 || {
  echo "ms-hub is required: python -m pip install 'modelscope-hub>=0.3.0'" >&2
  exit 2
}

mkdir -p "${ASSET_ROOT}/models/Qwen3.5-4B" "${ASSET_ROOT}/datasets" "${ASSET_ROOT}/checkpoints"
if [[ ! -s "${ASSET_ROOT}/models/Qwen3.5-4B/config.json" ]]; then
  ms-hub download "${BASE_ID}" --repo-type model --local-dir "${ASSET_ROOT}/models/Qwen3.5-4B"
fi
if [[ ! -s "${ASSET_ROOT}/datasets/ASSET_MANIFEST.json" ]]; then
  ms-hub download "${DATA_ID}" --repo-type dataset --local-dir "${ASSET_ROOT}/datasets"
fi
if [[ ! -d "${ASSET_ROOT}/datasets/images" ]]; then
  RS_MLLM_ASSET_ROOT="${ASSET_ROOT}" \
    "${PACKAGE_ROOT}/expert_lora/scripts/download_official_datasets.sh"
fi
if [[ ! -s "${ASSET_ROOT}/checkpoints/ASSET_MANIFEST.json" ]]; then
  ms-hub download "${CHECKPOINT_ID}" --repo-type model --local-dir "${ASSET_ROOT}/checkpoints"
fi

echo "Distillation assets ready under ${ASSET_ROOT}"
