#!/bin/bash
#SBATCH --job-name=vl2s --output=logs/viz_l2_single_%j.out --error=logs/viz_l2_single_%j.err
#SBATCH --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=1:00:00
set -e; export PYTHONUNBUFFERED=1
export DATA_ROOT=/home/u2024311149/RS-MLLM/datasets/shared_datasets
export PATH="/home/u2024311149/.opencode/bin:/home/u2024311149/.linuxbrew/bin:/home/u2024311149/.linuxbrew/sbin:/home/u2024311149/.conda/envs/rs_mllm/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/opt/app/spack/bin:/opt/app/spack/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
cd /home/u2024311149/RS-MLLM
echo "=== Single-image L2 pruning visualization ==="
echo "Date: $(date)"
python3 scripts/viz_l2_single.py \
  --image /home/u2024311149/RS-MLLM/datasets/shared_datasets/XLRS-Bench-lite/images_resized/xlrs_00003.png \
  --r 0.5 \
  --out_dir prune/output/viz_l2/single
echo "=== Done: $(date) ==="
