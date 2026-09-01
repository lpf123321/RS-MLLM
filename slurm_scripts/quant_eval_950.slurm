#!/bin/bash
#SBATCH --job-name=quant950
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
# 量化模型 950 清单评测（与 bf16 基线相同配置，仓库内评测器）
# 用法: sbatch slurm_scripts/quant_eval_950.slurm <model_dir> <profile> <outdir> [manifest]
set -euo pipefail

# 仓库根：clone 后设置环境变量 RSMLLM_REPO 或修改下方默认值
PROJECT=${RSMLLM_REPO:-/home/u2024311009/rs-mllm}
# 评测器（仓库内拆包目录脚本）
EVAL_DIR=${PROJECT}/evaluation/vllm_eval
# vllm 0.26 环境（uv 项目，clone 后 uv sync）
ENV=${RSMLLM_ENV:-/home/u2024311009/rs-mllm-uv-verify}
UV=${HOME}/.local/bin/uv
cd "${PROJECT}"

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${EVAL_DIR}${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${1:?用法: quant_eval_950.slurm <model_dir> <profile> <outdir> [manifest]}"
PROFILE="${2:?profile: mmerestore_bf16|mmerestore_w8a8|mmerestore_gptq}"
OUTDIR="${3:?outdir}"
MANIFEST="${4:-${RSMLLM_TILED:-/home/u2024311009/tiled_rag}/e0_pair/e2_manifest.jsonl}"

echo "=== quant950 start $(date -Is) model=$MODEL profile=$PROFILE ==="
"${UV}" run --project "${ENV}" --offline python "${EVAL_DIR}/vision_opd_vllm_eval.py" \
  --manifest "${MANIFEST}" \
  --model "$MODEL" --model-profile "$PROFILE" --output-dir "$OUTDIR" \
  --batch-size 64 --max-num-seqs 64 --max-model-len 16384

# 评分结果自动写入 $OUTDIR/clean_summary.json（评测器内置计分）
echo "=== quant950 done $(date -Is) ==="
