#!/bin/bash
#SBATCH --job-name=rs_mllm_eval
#SBATCH --output=evaluation/logs/eval_%j.out
#SBATCH --error=evaluation/logs/eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

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
echo "  PyTorch CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo "============================================"

# ===== Configurable =====
MODEL_PATH="$REPO_ROOT/models/Qwen3.5-2B"
DATASETS="all"
MAX_SAMPLES=0
DEVICE="cuda"
OUTPUT="$REPO_ROOT/evaluation/results.json"
EVAL_BATCH_SIZE=4 #
COMPILE_MODEL=""

# Override from command line args (e.g. --datasets vrsbench levircc)
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)       MODEL_PATH="$2";        shift 2 ;;
        --datasets)         DATASETS="$2";          shift 2 ;;
        --max_samples)      MAX_SAMPLES="$2";       shift 2 ;;
        --device)           DEVICE="$2";            shift 2 ;;
        --output)           OUTPUT="$2";            shift 2 ;;
        --eval_batch_size)  EVAL_BATCH_SIZE="$2";   shift 2 ;;
        --compile_model)    COMPILE_MODEL="--compile_model"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "  Model:          $MODEL_PATH"
echo "  Datasets:       $DATASETS"
echo "  Max samples:    $MAX_SAMPLES"
echo "  Device:         $DEVICE"
echo "  Output:         $OUTPUT"
echo "  Eval batch size: $EVAL_BATCH_SIZE"
echo "  Compile model:  $([ -n \"$COMPILE_MODEL\" ] && echo yes || echo no)"
echo "============================================"

cd "$REPO_ROOT"
python -m evaluation.main \
    --model_path "$MODEL_PATH" \
    --datasets $DATASETS \
    --max_samples $MAX_SAMPLES \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    --eval_batch_size $EVAL_BATCH_SIZE \
    $COMPILE_MODEL \
    "${EXTRA_ARGS[@]}"

echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
