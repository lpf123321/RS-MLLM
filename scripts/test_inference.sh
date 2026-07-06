#!/bin/bash
#SBATCH --job-name=test_qwen3_vl
#SBATCH --output=logs/test_inference_%j.out
#SBATCH --error=logs/test_inference_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

source activate rs_mllm

python -u scripts/test_inference.py
