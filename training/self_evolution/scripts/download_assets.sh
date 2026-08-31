#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ASSET_ROOT="${RS_MLLM_ASSET_ROOT:-${PACKAGE_ROOT}/.artifacts}"

QWEN_MODEL_ID="${RS_MLLM_QWEN3VL2B_MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
QWEN_REVISION="${RS_MLLM_QWEN3VL2B_REVISION:-master}"
SAM_MODEL_ID="${RS_MLLM_SAM3_MODEL_ID:-facebook/sam3}"
SAM_REVISION="${RS_MLLM_SAM3_REVISION:-master}"
DATASET_ID="${RS_MLLM_SELF_EVOLUTION_DATASET_ID:-Uchitachi/RS-MLLM-Self-Evolution-Data}"
DATASET_REVISION="${RS_MLLM_SELF_EVOLUTION_DATASET_REVISION:-master}"
ADAPTER_ID="${RS_MLLM_SELF_EVOLUTION_ADAPTER_ID:-Uchitachi/RS-MLLM-Self-Evolution-Checkpoints}"
ADAPTER_REVISION="${RS_MLLM_SELF_EVOLUTION_ADAPTER_REVISION:-master}"

command -v ms-hub >/dev/null 2>&1 || {
  echo "ms-hub is required: python -m pip install 'modelscope-hub>=0.3.0'" >&2
  exit 2
}

mkdir -p "${ASSET_ROOT}/models" "${ASSET_ROOT}/datasets"

if [[ ! -s "${ASSET_ROOT}/models/Qwen3-VL-2B-Instruct/config.json" ]]; then
  mkdir -p "${ASSET_ROOT}/models/Qwen3-VL-2B-Instruct"
  ms-hub download "${QWEN_MODEL_ID}" --repo-type model \
    --revision "${QWEN_REVISION}" \
    --local-dir "${ASSET_ROOT}/models/Qwen3-VL-2B-Instruct"
fi

if [[ ! -s "${ASSET_ROOT}/models/SAM3/sam3.pt" ]]; then
  mkdir -p "${ASSET_ROOT}/models/SAM3"
  ms-hub download "${SAM_MODEL_ID}" --repo-type model \
    --revision "${SAM_REVISION}" \
    --local-dir "${ASSET_ROOT}/models/SAM3"
fi

if [[ ! -s "${ASSET_ROOT}/datasets/self_evolution/ASSET_MANIFEST.json" ]]; then
  [[ -n "${DATASET_ID}" ]] || {
    echo "Set RS_MLLM_SELF_EVOLUTION_DATASET_ID after the dataset is published." >&2
    exit 2
  }
  mkdir -p "${ASSET_ROOT}/datasets/self_evolution"
  ms-hub download "${DATASET_ID}" --repo-type dataset \
    --revision "${DATASET_REVISION}" \
    --local-dir "${ASSET_ROOT}/datasets/self_evolution"
fi

# Reference adapters are optional: end-to-end runs train them from scratch.
if [[ -n "${ADAPTER_ID}" && ! -s "${ASSET_ROOT}/reference_adapters/ASSET_MANIFEST.json" ]]; then
  mkdir -p "${ASSET_ROOT}/reference_adapters"
  ms-hub download "${ADAPTER_ID}" --repo-type model \
    --revision "${ADAPTER_REVISION}" \
    --local-dir "${ASSET_ROOT}/reference_adapters"
fi

echo "Self-evolution assets ready under ${ASSET_ROOT}"
