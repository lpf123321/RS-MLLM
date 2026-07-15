#!/bin/bash
#SBATCH --job-name=q35_rs_bench
#SBATCH --output=logs/qwen35_rs_bench_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
ENV_DIR=${PROJECT}/environments/qwen35-finetune
MODEL=${PROJECT}/.ms_cache/models/Qwen--Qwen3.5-4B/snapshots/master
DATA=${DATA:-${PROJECT}/finetune_data/qwen35_rs_pilot_v2/validation.jsonl}
MODE=${MODE:?Set MODE=base or adapter}
SAMPLES=${SAMPLES:-100}

case "${MODE}" in
  base)
    MODEL_ARGS=(--model "${MODEL}")
    ;;
  adapter)
    if [[ -n "${ADAPTER_DIR:-}" ]]; then
      CHECKPOINT=${ADAPTER_DIR}
    else
      OUTPUT_PARENT=$(cat "${PROJECT}/results/qwen35_rs_lora_latest_pilot.txt")
      CHECKPOINT=$(find "${OUTPUT_PARENT}" -type d -name 'checkpoint-*' -print | sort -V | tail -1)
    fi
    [[ -d "${CHECKPOINT}" ]] || { echo "Adapter checkpoint not found: ${CHECKPOINT}" >&2; exit 2; }
    MODEL_ARGS=(--adapters "${CHECKPOINT}")
    ;;
  *)
    echo "MODE must be base or adapter, got: ${MODE}" >&2
    exit 2
    ;;
esac

RESULT=${PROJECT}/results/qwen35_rs_benchmark_${MODE}_${SAMPLES}_${SLURM_JOB_ID}.jsonl

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=1024
export FPS_MAX_FRAMES=16

cd "${PROJECT}"
echo "=== QWEN3.5 RS BENCHMARK START $(date -Is) ==="
echo "mode=${MODE} samples=${SAMPLES} result=${RESULT}"
(cd "${MODEL}" && sha256sum -c "${PROJECT}/model_manifests/qwen35-4b.sha256")

"${ENV_DIR}/.venv/bin/swift" infer \
  "${MODEL_ARGS[@]}" \
  --attn_impl sdpa \
  --infer_backend transformers \
  --val_dataset "${DATA}" \
  --split_dataset_ratio 0 \
  --val_dataset_sample "${SAMPLES}" \
  --max_batch_size 1 \
  --max_new_tokens 128 \
  --temperature 0 \
  --stream false \
  --result_path "${RESULT}"

echo "=== QWEN3.5 RS BENCHMARK END $(date -Is) ==="
wc -l "${RESULT}"
