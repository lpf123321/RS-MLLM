#!/bin/bash
#SBATCH --job-name=rsmllm-docker-build
#SBATCH --output=/home/u2024311009/RS-MLLM/logs/docker_build_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# 在 GPU 计算节点上: docker 存在性检查 -> nvidia-smi -> 构建镜像 -> 容器内 import 验证
set -euo pipefail
cd /home/u2024311009/rs-mllm-docker

echo "== docker =="; docker --version
echo "== gpu =="; nvidia-smi -L || true

echo "== build =="
docker build -t rs-mllm . 2>&1 | tail -5

echo "== 容器内 import 验证 =="
docker run --rm rs-mllm python -c \
  "import torch, transformers; print('TORCH_OK', torch.__version__, torch.cuda.is_available(), '| transformers', transformers.__version__)"

echo "DOCKER_BUILD_VERIFY_DONE"
