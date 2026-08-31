#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${1:-}"
STAGING_ROOT="${2:-}"
ACTION="${3:-upload}"
VISIBILITY="${RS_MLLM_MODELSCOPE_VISIBILITY:-private}"
ONLY="${4:-all}"

[[ -n "${NAMESPACE}" && -n "${STAGING_ROOT}" ]] || {
  echo "Usage: $0 <namespace> <staging-root> [create|upload] [all|staging-directory]" >&2
  exit 2
}
[[ "${NAMESPACE}" != */* ]] || { echo "Namespace must not contain '/'." >&2; exit 2; }
[[ "${ACTION}" == create || "${ACTION}" == upload ]] || {
  echo "Action must be create or upload." >&2
  exit 2
}
[[ "${VISIBILITY}" == private || "${VISIBILITY}" == internal ]] || {
  echo "Only private/internal upload is allowed for this reproducibility release." >&2
  exit 2
}
[[ "${ONLY}" == all || "${ONLY}" == distillation_data || \
   "${ONLY}" == self_evolution_data || "${ONLY}" == expert_checkpoints || \
   "${ONLY}" == distillation_checkpoints || "${ONLY}" == self_evolution_checkpoints ]] || {
  echo "Unknown staging directory selector: ${ONLY}" >&2
  exit 2
}

command -v ms-hub >/dev/null 2>&1 || {
  echo "Install the official client: python -m pip install 'modelscope-hub>=0.2.0'" >&2
  exit 2
}

# Authentication is deliberately interactive/persisted. Never put a token in
# this script, a command-line argument, a log, or the Git repository.
ms-hub whoami >/dev/null

DATASET_DISTILLATION="${NAMESPACE}/RS-MLLM-Distillation-Data"
DATASET_SELF_EVOLUTION="${NAMESPACE}/RS-MLLM-Self-Evolution-Data"
MODEL_EXPERT="${NAMESPACE}/RS-MLLM-Expert-Checkpoints"
MODEL_DISTILLATION="${NAMESPACE}/RS-MLLM-Distillation-Checkpoints"
MODEL_SELF_EVOLUTION="${NAMESPACE}/RS-MLLM-Self-Evolution-Checkpoints"

if [[ "${ONLY}" == all || "${ONLY}" == distillation_data ]]; then
  ms-hub create "${DATASET_DISTILLATION}" --repo-type dataset --visibility "${VISIBILITY}" \
    --license other --description "RS-MLLM retained Expert annotations and upstream image hashes" --exist-ok
fi
if [[ "${ONLY}" == all || "${ONLY}" == self_evolution_data ]]; then
  ms-hub create "${DATASET_SELF_EVOLUTION}" --repo-type dataset --visibility "${VISIBILITY}" \
    --license apache-2.0 --description "RS-MLLM CVSearch OPSD self-evolution data" --exist-ok
fi
if [[ "${ONLY}" == all || "${ONLY}" == expert_checkpoints ]]; then
  ms-hub create "${MODEL_EXPERT}" --repo-type model --visibility "${VISIBILITY}" \
    --license apache-2.0 --description "RS-MLLM General/Grounding Expert deltas and LoRA adapters" --exist-ok
fi
if [[ "${ONLY}" == all || "${ONLY}" == distillation_checkpoints ]]; then
  ms-hub create "${MODEL_DISTILLATION}" --repo-type model --visibility "${VISIBILITY}" \
    --license other --description "RS-MLLM SFT, OPD, and comparison adapters" --exist-ok
fi
if [[ "${ONLY}" == all || "${ONLY}" == self_evolution_checkpoints ]]; then
  ms-hub create "${MODEL_SELF_EVOLUTION}" --repo-type model --visibility "${VISIBILITY}" \
    --license other --description "RS-MLLM Qwen3-VL-2B and SAM3 self-evolution checkpoints" --exist-ok
fi

if [[ "${ACTION}" == create ]]; then
  echo "Repositories created; no files uploaded."
  exit 0
fi

declare -a SPECS=(
  "dataset|${DATASET_DISTILLATION}|distillation_data"
  "dataset|${DATASET_SELF_EVOLUTION}|self_evolution_data"
  "model|${MODEL_EXPERT}|expert_checkpoints"
  "model|${MODEL_DISTILLATION}|distillation_checkpoints"
  "model|${MODEL_SELF_EVOLUTION}|self_evolution_checkpoints"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r repo_type repo_id directory <<<"${spec}"
  if [[ "${ONLY}" != all && "${ONLY}" != "${directory}" ]]; then
    continue
  fi
  local_path="${STAGING_ROOT}/${directory}"
  [[ -s "${local_path}/ASSET_MANIFEST.json" && -s "${local_path}/SHA256SUMS" ]] || {
    echo "Staging directory is incomplete: ${local_path}" >&2
    exit 2
  }
  ms-hub upload "${repo_id}" "${local_path}" --repo-type "${repo_type}" \
    --revision master --max-workers 8 \
    --commit-message "Publish retained RS-MLLM reproducibility assets"
done

echo "Requested ModelScope repositories uploaded successfully: ${ONLY}"
