#!/bin/bash
#SBATCH --job-name=rs_mllm_eval_qwen35vl
#SBATCH --output=evaluation/logs/eval_qwen35vl_%j.out
#SBATCH --error=evaluation/logs/eval_qwen35vl_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -e
export PYTHONUNBUFFERED=1

REPO_ROOT="$SLURM_SUBMIT_DIR"
LOG_DIR="$REPO_ROOT/evaluation/logs"
mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets}"

echo "============================================"
echo "  SLURM Job ID:     $SLURM_JOB_ID"
echo "  SLURM Node:       $SLURM_NODELIST"
echo "  GPUs Allocated:   $CUDA_VISIBLE_DEVICES"
echo "  Node:             $(hostname)"
echo "============================================"

export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:/opt/app/anaconda3/bin:$PATH"
pip install -q nltk rouge-score 2>/dev/null || true

echo "  Conda env:        $CONDA_DEFAULT_ENV"
echo "  Python:           $(which python)"
echo "  PyTorch CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo "============================================"

# ===== Configurable — Qwen3.5-VL-4B defaults =====
MODEL_PATH="/home/u2024311149/models/Qwen3.5-4B"
DATASETS="all"
MAX_SAMPLES=0
START_OFFSET=0
DEVICE="cuda"
OUTPUT="$REPO_ROOT/evaluation/results_qwen35vl_4b.json"
EVAL_BATCH_SIZE=4
COMPILE_MODEL="--compile_model"
DISABLE_THINKING="--disable_thinking"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)       MODEL_PATH="$2";        shift 2 ;;
        --datasets)
            DATASETS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS="$DATASETS $1"
                shift
            done
            DATASETS="${DATASETS# }"
            ;;
        --max_samples)      MAX_SAMPLES="$2";       shift 2 ;;
        --start_offset)     START_OFFSET="$2";     shift 2 ;;
        --device)           DEVICE="$2";            shift 2 ;;
        --output)           OUTPUT="$2";            shift 2 ;;
        --eval_batch_size)  EVAL_BATCH_SIZE="$2";   shift 2 ;;
        --no_compile)       COMPILE_MODEL="";       shift ;;
        --enable_thinking)  DISABLE_THINKING="--enable_thinking"; shift ;;
        *) shift ;;
    esac
done

echo "  Model:          $MODEL_PATH"
echo "  Datasets:       $DATASETS"
echo "  Max samples:    $MAX_SAMPLES"
echo "  Start offset:   $START_OFFSET"
echo "  Device:         $DEVICE"
echo "  Output:         $OUTPUT"
echo "  Eval batch size: $EVAL_BATCH_SIZE"
echo "  Compile model:  $([ -n "$COMPILE_MODEL" ] && echo yes || echo no)"
echo "  Disable thinking: $([ "$DISABLE_THINKING" = "--disable_thinking" ] && echo yes || echo no)"
echo "============================================"

cd "$REPO_ROOT"
python -m evaluation.main_qwen35vl \
    --model_path "$MODEL_PATH" \
    --datasets $DATASETS \
    --max_samples $MAX_SAMPLES \
    --start_offset $START_OFFSET \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    --eval_batch_size $EVAL_BATCH_SIZE \
    $COMPILE_MODEL \
    $DISABLE_THINKING

echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
