#!/bin/bash
#SBATCH --job-name=e0_pair_v2
#SBATCH --output=/home/u2024311009/tiled_rag/e0_pair_v2_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a30
#SBATCH --time=02:00:00
set -euo pipefail
ROOT=/home/u2024311009/tiled_rag
PY=/home/u2024311009/rs-mllm-dataset-fixes/.venv/bin/python
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
MANIFEST=/home/u2024311009/rs-mllm-dataset-fixes/results/testset_0_1pct_v2/testset.jsonl
MODEL=/home/u2024311009/tiled_rag/model_mmerestore
mkdir -p "${ROOT}/e0_pair"
echo "=== E0 V2 PAIR START $(date -Is) ==="
"${PY}" -m rs_mllm_dataset_fixes.vision_opd_eval \
  --manifest "${MANIFEST}" --model "${MODEL}" \
  --output-dir "${ROOT}/e0_pair/v2_bf16" --quantization bf16
echo "--- bf16 pass done $(date -Is)"
"${PY}" -m rs_mllm_dataset_fixes.vision_opd_eval \
  --manifest "${MANIFEST}" --model "${MODEL}" \
  --output-dir "${ROOT}/e0_pair/v2_int8" --quantization int8
echo "--- int8 pass done $(date -Is)"
echo "=== E0 V2 PAIR DONE $(date -Is) ==="
