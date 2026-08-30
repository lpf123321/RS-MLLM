#!/bin/bash
# Smoke test all task-to-expert routes with the candidate pruning policy.
#
#SBATCH --job-name=task_router_smoke
#SBATCH --output=logs/task_router_smoke_%j.out
#SBATCH --error=logs/task_router_smoke_%j.err
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/u2024311149/RS-MLLM
export DATA_ROOT=/users/u2024311136/shared/shared_datasets
export PYTHONPATH=/home/u2024311149/RS-MLLM
export PATH="/home/u2024311149/.conda/envs/rs_mllm/bin:$PATH"

python scripts/smoke_test_delta_task_router.py

echo "=== DELTA TASK ROUTER SMOKE DONE ==="
