#!/bin/bash
# ============================================================
# RS-MLLM 四专家训练一键启动（统一 SFT 主干 + 专家 / 续训）
#
# 用法:
#   bash scripts/train.sh stage1           # 统一 SFT 主干（Qwen3.5-4B from-base）
#   bash scripts/train.sh expert_general   # General 专家（v2, from-base）
#   bash scripts/train.sh expert_grounding # Grounding 专家（from-base, 语义对照）
#   bash scripts/train.sh expert_change    # Change 专家（from-base, 语义对照）
#   bash scripts/train.sh ga2_general      # General 续训（防遗忘, 主干起点）
#   bash scripts/train.sh a1_grounding     # Grounding 域对齐续训（2048）
#   bash scripts/train.sh a2b_change       # Change 续训（防遗忘 v2）
#   bash scripts/train.sh caption          # Caption 专家（GA2 起点, 双域）
#   bash scripts/train.sh all              # 完整流水线（见顺序说明）
#   bash scripts/train.sh dry [stage]      # 只打印 sbatch 命令不提交
# ============================================================
set -e

STAGE="${1:-}"
DRY=0
if [ "$STAGE" = "dry" ]; then
  DRY=1; STAGE="${2:-}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/training" && pwd)"
SLURM_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
declare -A SLURM=(
  [stage1]="sft_stage1.slurm"
  [stage1_clean]="sft_stage1_clean.slurm"
  [stage2]="sft_stage2.slurm"
  [expert_general]="sft_expert_general.slurm"
  [expert_grounding]="sft_expert_grounding.slurm"
  [expert_change]="sft_expert_change.slurm"
  [ga2_general]="sft_ga2_general.slurm"
  [ga3_general]="sft_ga3_general.slurm"
  [a1_grounding]="sft_a1_grounding.slurm"
  [a2_change]="sft_a2_change.slurm"
  [a2b_change]="sft_a2b_change.slurm"
  [caption]="sft_caption_expert.slurm"
)

usage() {
  echo "可用 stage: stage1 stage1_clean stage2 expert_general expert_grounding expert_change"
  echo "            ga2_general ga3_general a1_grounding a2_change a2b_change caption all"
  echo "dry:  bash scripts/train.sh dry <stage>"
}

if [ "$STAGE" = "all" ]; then
  for s in stage1_clean ga2_general a1_grounding a2b_change caption; do
    echo ">>> 提交: $s"
    sbatch "$SLURM_DIR/${SLURM[$s]}"
  done
  exit 0
fi

if [ -z "$STAGE" ] || [ -z "${SLURM[$STAGE]:-}" ]; then
  usage; exit 1
fi

# --- 数据自检：关键训练 json 缺失时提示下载 ---
VRS="$REPO_ROOT/finetune_framework/VRSbench"
NEED_JSON=""
case "$STAGE" in
  stage1_clean|stage1)    NEED_JSON="manifest_sft_train.json combined_train.json" ;;
  expert_general|ga3_general) NEED_JSON="expert_data_v2/general_understanding.json" ;;
  expert_grounding)       NEED_JSON="expert_data/grounding.json" ;;
  expert_change|a2_change) NEED_JSON="expert_data/change.json" ;;
  a2b_change)             NEED_JSON="a2_change_mix.json" ;;
  a1_grounding|a1_full)   NEED_JSON="a1_domainalign.json" ;;
  ga2_general)            NEED_JSON="g_a2_mix.json" ;;
  caption)                NEED_JSON="expert_data_caption.jsonl" ;;
esac
MISSING=""
for j in $NEED_JSON; do
  [ -f "$VRS/$j" ] || MISSING="$MISSING $j"
done
if [ -n "$MISSING" ]; then
  echo "[train] 缺少训练数据:${MISSING}"
  echo "       请先运行:  bash scripts/fetch_training_data.sh   （从 ModelScope 下载清洗 json）"
  echo "       或按脚本注释手动放置数据到 $VRS"
  exit 1
fi

SBATCH_CMD="sbatch $SLURM_DIR/${SLURM[$STAGE]}"
echo ">>> $SBATCH_CMD"
if [ "$DRY" = "1" ]; then exit 0; fi
eval "$SBATCH_CMD"