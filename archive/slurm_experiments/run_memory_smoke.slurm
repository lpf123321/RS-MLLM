#!/bin/bash
# Submit both methods with: sbatch evaluation/run_memory_smoke.slurm
#SBATCH --job-name=token_mem_smoke
#SBATCH --output=evaluation/logs/token_mem_smoke_%A_%a.out
#SBATCH --error=evaluation/logs/token_mem_smoke_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-1

set -eo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/home/u2025310307/models/Qwen3.5-4B}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/datasets_data/mme_rs.jsonl}"
RESULTS_DIR="$REPO_ROOT/evaluation/results"
mkdir -p "$RESULTS_DIR" "$REPO_ROOT/evaluation/logs"

source /opt/app/anaconda3/etc/profile.d/conda.sh
conda activate rs_mllm
set -u

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        METHOD="baseline"
        KEEP_RATIO="1.0"
        PRUNER="random"
        ;;
    1)
        METHOD="random"
        KEEP_RATIO="0.5"
        PRUNER="random"
        ;;
    *)
        echo "Unexpected array task ID: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

echo "job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "method=${METHOD} model=${MODEL_PATH} data=${DATA_PATH}"

cd "$REPO_ROOT"
python -m evaluation.benchmark_memory \
    --model_path "$MODEL_PATH" \
    --dataset mme \
    --data_path "$DATA_PATH" \
    --method "$METHOD" \
    --pruner "$PRUNER" \
    --keep_ratio "$KEEP_RATIO" \
    --sample_count 20 \
    --sample_seed 2026 \
    --pruner_seed 2026 \
    --output "$RESULTS_DIR/memory_smoke_${METHOD}.json"
