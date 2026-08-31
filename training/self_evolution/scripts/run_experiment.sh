#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF_EVOLUTION_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd -- "${SELF_EVOLUTION_ROOT}/../.." && pwd)"
ASSET_ROOT="${RS_MLLM_ASSET_ROOT:-${SELF_EVOLUTION_ROOT}/.artifacts}"
OUTPUT_ROOT="${RS_MLLM_SELF_EVOLUTION_OUTPUT_ROOT:-${SELF_EVOLUTION_ROOT}/outputs}"
DATA_ROOT="${ASSET_ROOT}/datasets/self_evolution"

METHOD="${1:-}"
ACTION="${2:-doctor}"
case "${METHOD}" in
  tree|strict|fine|all) ;;
  *) echo "Usage: $0 {tree|strict|fine|all} {doctor|prepare|initial|smoke|full|retest}" >&2; exit 2 ;;
esac
case "${ACTION}" in
  doctor|prepare|initial|smoke|full|retest) ;;
  *) echo "Unknown action: ${ACTION}" >&2; exit 2 ;;
esac

if [[ "${METHOD}" == all ]]; then
  for current_method in tree strict fine; do
    "${BASH_SOURCE[0]}" "${current_method}" "${ACTION}"
  done
  exit 0
fi

[[ -s "${DATA_ROOT}/ASSET_MANIFEST.json" ]] || {
  echo "Assets are missing. Run ${SCRIPT_DIR}/download_assets.sh first." >&2
  exit 2
}

SAM_T0="${SAM_T0:-$(find -L "${ASSET_ROOT}/models/SAM3" -type f -name 'sam3.pt' -print -quit)}"
[[ -n "${SAM_T0}" && -s "${SAM_T0}" ]] || { echo "Cannot locate stock sam3.pt" >&2; exit 2; }

export SELF_EVOLUTION_ROOT
export SELF_EVOLUTION_WORKSPACE="${SELF_EVOLUTION_WORKSPACE:-${OUTPUT_ROOT}/${METHOD}}"
export VISION_OPD_ROOT="${VISION_OPD_ROOT:-${REPOSITORY_ROOT}/training/distillation/opsd/vision_opd}"
export VOPD_DATA_ROOT="${VOPD_DATA_ROOT:-${DATA_ROOT}/vision_opd}"
export VOPD_PREPARED_SPLITS="${VOPD_PREPARED_SPLITS:-${VOPD_DATA_ROOT}/splits}"
export MLLM_T0="${MLLM_T0:-${ASSET_ROOT}/models/Qwen3-VL-2B-Instruct}"
export SAM_T0
export SPACY_MODEL="${SPACY_MODEL:-${DATA_ROOT}/runtime/en_core_web_sm-3.8.0}"
export SAM_CONDA_ENV="${SAM_CONDA_ENV:-sam3_lora}"
export OPD_CONDA_ENV="${OPD_CONDA_ENV:-vision_opd}"
ORCHESTRATOR_CONDA_ENV="${ORCHESTRATOR_CONDA_ENV:-cvsearch}"

case "${METHOD}" in
  tree)
    CONFIG="${SELF_EVOLUTION_ROOT}/orchestrator/configs/repro_tree_node_weak_box.yaml"
    MANIFEST_DIR="${DATA_ROOT}/frozen_negatives/tree_node_weak_box"
    ;;
  strict)
    CONFIG="${SELF_EVOLUTION_ROOT}/orchestrator/configs/repro_strict_local_recovery.yaml"
    MANIFEST_DIR="${DATA_ROOT}/frozen_negatives/strict_local_recovery"
    ;;
  fine)
    CONFIG="${SELF_EVOLUTION_ROOT}/orchestrator/configs/repro_fine_trace_only.yaml"
    MANIFEST_DIR="${DATA_ROOT}/frozen_negatives/fine_trace_only"
    export STRICT_MLLM_SMOKE="${STRICT_MLLM_SMOKE:-${OUTPUT_ROOT}/strict/models/smoke/round_00/mllm_candidate}"
    export STRICT_MLLM_FULL="${STRICT_MLLM_FULL:-${OUTPUT_ROOT}/strict/models/full/round_00/mllm_candidate}"
    ;;
esac
export SAM_TRAIN_NEGATIVE_MANIFEST="${SAM_TRAIN_NEGATIVE_MANIFEST:-${MANIFEST_DIR}/train.jsonl}"
export SAM_VALID_NEGATIVE_MANIFEST="${SAM_VALID_NEGATIVE_MANIFEST:-${MANIFEST_DIR}/valid.jsonl}"

RUN=(conda run --no-capture-output -n "${ORCHESTRATOR_CONDA_ENV}" python "${SELF_EVOLUTION_ROOT}/self_evolution.py" --config "${CONFIG}")

case "${ACTION}" in
  doctor)
    "${RUN[@]}" doctor
    ;;
  prepare)
    "${RUN[@]}" prepare
    ;;
  initial)
    "${RUN[@]}" prepare
    "${RUN[@]}" evaluate --split valid --checkpoint t0 --stage full
    ;;
  smoke)
    "${RUN[@]}" prepare
    "${RUN[@]}" smoke --round 0
    ;;
  full)
    "${RUN[@]}" prepare
    "${RUN[@]}" run-loop --stage full --max-rounds 1
    ;;
  retest)
    "${RUN[@]}" evaluate --split valid --checkpoint best --stage full
    ;;
esac
