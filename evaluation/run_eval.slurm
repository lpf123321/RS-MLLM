#!/bin/bash
#SBATCH --job-name=rs_mllm_eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

export PATH="/opt/app/anaconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate rs_mllm
pip install -q nltk rouge-score 2>/dev/null || true

MODEL_PATH="$REPO_ROOT/models/Qwen3-VL-2B-Instruct"
DATASETS="all"
MAX_SAMPLES=0
DEVICE="cuda"
OUTPUT="$REPO_ROOT/evaluation/results.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)   MODEL_PATH="$2";    shift 2 ;;
        --datasets)     DATASETS="$2";      shift 2 ;;
        --max_samples)  MAX_SAMPLES="$2";   shift 2 ;;
        --device)       DEVICE="$2";        shift 2 ;;
        --output)       OUTPUT="$2";        shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  SLURM Job ID:     $SLURM_JOB_ID"
echo "  Node:             $(hostname)"
echo "  GPU:              $CUDA_VISIBLE_DEVICES"
echo "  Conda env:        $CONDA_DEFAULT_ENV"
echo "  PyTorch CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"
echo "  Model:      $MODEL_PATH"
echo "  Datasets:   $DATASETS"
echo "  Max samples: $MAX_SAMPLES"
echo "  Device:     $DEVICE"
echo "  Output:     $OUTPUT"
echo "============================================"

mkdir -p "$(dirname "$OUTPUT")"

python -m evaluation.main \
    --model_path "$MODEL_PATH" \
    --datasets $DATASETS \
    --max_samples $MAX_SAMPLES \
    --device "$DEVICE" \
    --output "$OUTPUT"

echo "============================================"
echo "  Job finished: $(date)"
echo "============================================"
