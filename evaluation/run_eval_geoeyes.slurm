#!/bin/bash
#SBATCH --job-name=geoeyes_eval
#SBATCH --output=evaluation/logs/geoeyes_eval_%j.out
#SBATCH --error=evaluation/logs/geoeyes_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

set -e
export PYTHONUNBUFFERED=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
LOG_DIR="$REPO_ROOT/evaluation/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  SLURM Job ID:     $SLURM_JOB_ID"
echo "  SLURM Node:       $SLURM_NODELIST"
echo "  GPUs Allocated:   $CUDA_VISIBLE_DEVICES"
echo "  Node:             $(hostname)"
echo "============================================"

source /opt/app/anaconda3/etc/profile.d/conda.sh
conda activate rs_mllm

echo "  Conda env:        $CONDA_DEFAULT_ENV"
echo "  Python:           $(which python)"
echo "  PyTorch CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

MODEL_PATH="$REPO_ROOT/models/GeoEyes"
DATASETS="all"
MAX_SAMPLES=0
DEVICE="cuda"
OUTPUT="$REPO_ROOT/evaluation/results_geoeyes.json"
GPU_MEMORY_UTILIZATION=0.92

# Override from command line args
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)       MODEL_PATH="$2";        shift 2 ;;
        --datasets)         DATASETS="$2";          shift 2 ;;
        --max_samples)      MAX_SAMPLES="$2";       shift 2 ;;
        --device)           DEVICE="$2";            shift 2 ;;
        --output)           OUTPUT="$2";            shift 2 ;;
        --gpu_memory)       GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "  Model:          $MODEL_PATH"
echo "  Datasets:       $DATASETS"
echo "  Max samples:    $MAX_SAMPLES"
echo "  Device:         $DEVICE"
echo "  Output:         $OUTPUT"
echo "  GPU memory:     $GPU_MEMORY_UTILIZATION"
echo "============================================"

cd "$REPO_ROOT"
python -m evaluation.main \
    --adapter geoeyes \
    --model_path "$MODEL_PATH" \
    --datasets $DATASETS \
    --max_samples $MAX_SAMPLES \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    --eval_batch_size 1 \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --vllm_max_model_len 4096 \
    "${EXTRA_ARGS[@]}"

echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
