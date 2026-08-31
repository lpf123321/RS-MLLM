#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ASSET_ROOT="${RS_MLLM_ASSET_ROOT:-${PACKAGE_ROOT}/.artifacts}"

BASE_MODEL_ID="${RS_MLLM_QWEN35_MODEL_ID:-Qwen/Qwen3.5-4B}"
BASE_MODEL_REVISION="${RS_MLLM_QWEN35_REVISION:-master}"
DATASET_ID="${RS_MLLM_DISTILLATION_DATASET_ID:-Uchitachi/RS-MLLM-Distillation-Data}"
DATASET_REVISION="${RS_MLLM_DISTILLATION_DATASET_REVISION:-master}"
EXPERT_MODEL_ID="${RS_MLLM_EXPERT_MODEL_ID:-Uchitachi/RS-MLLM-Expert-Checkpoints}"
EXPERT_MODEL_REVISION="${RS_MLLM_EXPERT_MODEL_REVISION:-master}"

command -v ms-hub >/dev/null 2>&1 || {
  echo "ms-hub is required: python -m pip install 'modelscope-hub>=0.3.0'" >&2
  exit 2
}

mkdir -p "${ASSET_ROOT}"

if [[ ! -s "${ASSET_ROOT}/base/Qwen3.5-4B/config.json" ]]; then
  mkdir -p "${ASSET_ROOT}/base/Qwen3.5-4B"
  ms-hub download "${BASE_MODEL_ID}" --repo-type model \
    --revision "${BASE_MODEL_REVISION}" \
    --local-dir "${ASSET_ROOT}/base/Qwen3.5-4B"
fi

if [[ ! -s "${ASSET_ROOT}/datasets/ASSET_MANIFEST.json" ]]; then
  [[ -n "${DATASET_ID}" ]] || {
    echo "Set RS_MLLM_DISTILLATION_DATASET_ID after the dataset is published." >&2
    exit 2
  }
  mkdir -p "${ASSET_ROOT}/datasets"
  ms-hub download "${DATASET_ID}" --repo-type dataset \
    --revision "${DATASET_REVISION}" \
    --local-dir "${ASSET_ROOT}/datasets"
fi

if [[ ! -d "${ASSET_ROOT}/datasets/images" ]]; then
  "${SCRIPT_DIR}/download_official_datasets.sh"
fi

if [[ ! -s "${ASSET_ROOT}/expert_models/ASSET_MANIFEST.json" ]]; then
  [[ -n "${EXPERT_MODEL_ID}" ]] || {
    echo "Set RS_MLLM_EXPERT_MODEL_ID after the expert deltas are published." >&2
    exit 2
  }
  mkdir -p "${ASSET_ROOT}/expert_models"
  ms-hub download "${EXPERT_MODEL_ID}" --repo-type model \
    --revision "${EXPERT_MODEL_REVISION}" \
    --local-dir "${ASSET_ROOT}/expert_models"
fi

echo "Assets ready under ${ASSET_ROOT}"
