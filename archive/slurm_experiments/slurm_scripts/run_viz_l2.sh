#!/bin/bash
#SBATCH --job-name=vl2 --output=logs/viz_l2_%j.out --error=logs/viz_l2_%j.err
#SBATCH --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=1:00:00
set -e; export PYTHONUNBUFFERED=1
export DATA_ROOT=/home/u2024311149/RS-MLLM/datasets/shared_datasets
export PATH="/home/u2024311149/.opencode/bin:/home/u2024311149/.linuxbrew/bin:/home/u2024311149/.linuxbrew/sbin:/home/u2024311149/.conda/envs/rs_mllm/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/opt/app/spack/bin:/opt/app/spack/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
cd /home/u2024311149/RS-MLLM
echo "=== L2 pruning visualization ==="
echo "Date: $(date)"
python3 scripts/viz_l2.py --r 0.5 --n_samples 4
echo "=== Done: $(date) ==="
