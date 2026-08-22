#!/bin/bash
#SBATCH --job-name=delta_fourier
#SBATCH --output=logs/delta_fourier_%j.out
#SBATCH --error=logs/delta_fourier_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export DATA_ROOT=/users/u2024311136/shared/shared_datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"
python scripts/smoke_test_delta_fourier.py
