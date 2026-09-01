#!/bin/bash
# ============================================================
# RS-MLLM 四专家训练一键启动（统一 SFT 主干 + 专家 / 续训）
#
# 支持:
#   - 单阶段提交      bash scripts/train.sh stage1_clean
#   - dry 预览        bash scripts/train.sh dry stage1_clean
#   - 本地直接运行    bash scripts/train.sh --local stage1_clean   (无需 sbatch)
#   - 完整流水线      bash scripts/train.sh all                      (SLURM afterok 依赖链)
#   - 本地串联流水线  bash scripts/train.sh --local all              (顺序前台运行)
#
# 每个训练阶段结束后会自动安排 LoRA->merged 合并，满足后续续训起点。
# 数据缺失时提示先运行 scripts/fetch_training_data.sh。
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="$REPO_ROOT/scripts/training"
VRS="$REPO_ROOT/finetune_framework/VRSbench"

# 解析参数：默认本地前台直跑；--slurm 显式提交 SLURM（需集群）；--local 兼容旧写法(等价默认)
LOCAL=0
ARGS=()
for a in "$@"; do
  [ "$a" = "--slurm" ] && { LOCAL=1; continue; }
  [ "$a" = "--local" ] && continue
  ARGS+=("$a")
done
STAGE="${ARGS[0]:-}"
DRY=0
if [ "$STAGE" = "dry" ]; then DRY=1; STAGE="${ARGS[1]:-}"; fi

# 阶段配置：slurm | 所需数据 | 合并命令(lora_ckpt|merged_name)
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
declare -A NEED=(
  [stage1]=combined_train.json
  [stage1_clean]=manifest_sft_train.json
  [stage2]=combined_train.json
  [expert_general]=expert_data_v2/general_understanding.json
  [expert_grounding]=expert_data/grounding.json
  [expert_change]=expert_data/change.json
  [ga2_general]=g_a2_mix.json
  [ga3_general]=expert_data_v2/general_understanding.json
  [a1_grounding]=a1_domainalign.json
  [a2_change]=expert_data/change.json
  [a2b_change]=a2_change_mix.json
  [caption]=expert_data_caption.jsonl
)
# 合并映射: <stage>=<lora_ckpt>|<merged_name>   （空 = 该阶段不需要合并即可供后续使用）
declare -A MERGE=(
  [stage1_clean]="outputs/checkpoints/sft_stage1_clean|sft_stage1_clean"
  [expert_general]="outputs/checkpoints/expert_general_v2|expert_general_v2"
  [ga2_general]="outputs/checkpoints/ga2_general|ga2_general_checkpoint-451"
  [a1_grounding]="outputs/checkpoints/a1_grounding|a1_grounding_checkpoint-latest"
  [a2b_change]="outputs/checkpoints/a2b_change|a2b_change_checkpoint-latest"
  [caption]="outputs/checkpoints/caption_expert|caption_expert_checkpoint-latest"
)

usage() {
  echo "用法: bash scripts/train.sh <stage|all>   (默认本地前台直跑; --slurm 提交集群; dry: 预览)"
  echo "stages: stage1 stage1_clean stage2 expert_general expert_grounding expert_change"
  echo "        ga2_general ga3_general a1_grounding a2_change a2b_change caption all"
  echo "--slurm: 显式提交 SLURM 调度器(需集群; 默认本地直跑不依赖 SLURM)"
}

check_data() {
  local k="$1" miss="" base
  for j in ${NEED[$k]}; do
    base="finetune_framework/VRSbench"
    # 处理子路径
    [ -f "$REPO_ROOT/$base/$j" ] || miss="$miss $j"
  done
  if [ -n "$miss" ]; then
    echo "[train] 缺少数据:$miss"
    echo "       请先:  bash scripts/fetch_training_data.sh   （ModelScope 下载）"
    exit 1
  fi
}

run_slurm() {
  # 提交单阶段 + 依赖属性；若该阶段要合并则一并排队
  local stage="$1" dep="${2:-}" cmd="" depflag
  depflag=
  [ -n "$dep" ] && depflag="--dependency=afterok:$dep"
  sbatch $depflag "$SLURM_DIR/${SLURM[$stage]}"
}

run_local() {
  # 前台运行单阶段（继承 CONDA_HOME/TRAIN_DATA_ROOT 等环境变量）
  bash "$SLURM_DIR/${SLURM[$stage]}"
}

# --- 主逻辑 ---
if [ "$STAGE" = "all" ]; then
  if [ "$LOCAL" = "0" ]; then
    echo "== 本地串联流水线(默认; --slurm 走集群) =="
    for s in stage1_clean ga2_general a1_grounding a2b_change caption; do
      check_data "$s"
      echo ">>> 运行: $s (前台)"
      bash "$SLURM_DIR/${SLURM[$s]}"
      if [ -n "${MERGE[$s]:-}" ]; then
        IFS='|' read -r lora name <<< "${MERGE[$s]}"
        echo ">>> 合并: $s -> $name"
        bash "$REPO_ROOT/scripts/merge_checkpoint.sh" "$REPO_ROOT/$lora" "$name"
      fi
    done
    echo "== 流水线完成 =="
    exit 0
  fi

  echo "== SLURM 依赖链流水线(--slurm) =="
  JOB_IDS=()
  PREV=""
  for s in stage1_clean ga2_general a1_grounding a2b_change caption; do
    check_data "$s"
    # 合并前置步骤：先提交 merge？（此处简化为训练后自动合并）
    out=$(sbatch ${PREV:+--dependency=afterok:$PREV} "$SLURM_DIR/${SLURM[$s]}" 2>&1)
    jid=$(echo "$out" | grep -oE '[0-9]+' | tail -1)
    echo "  + $s  -> job $jid"
    PREV="$jid"
    if [ -n "${MERGE[$s]:-}" ]; then
      IFS='|' read -r lora name <<< "${MERGE[$s]}"
      mout=$(sbatch --dependency=afterok:$jid "$SLURM_DIR/merge.slurm" "$REPO_ROOT/$lora" "$name" 2>&1)
      mjid=$(echo "$mout" | grep -oE '[0-9]+' | tail -1)
      echo "    - merge $s -> job $mjid"
      PREV="$mjid"
    fi
  done
  echo "== 流水线已提交。依赖链: ${JOB_IDS[*]} =="
  exit 0
fi

if [ -z "$STAGE" ] || [ -z "${SLURM[$STAGE]:-}" ]; then
  usage; exit 1
fi

check_data "$STAGE"

if [ "$LOCAL" = "0" ]; then
  echo ">>> 本地运行: $STAGE"
  bash "$SLURM_DIR/${SLURM[$STAGE]}"
  if [ -n "${MERGE[$STAGE]:-}" ]; then
    IFS='|' read -r lora name <<< "${MERGE[$STAGE]}"
    echo ">>> 合并: $STAGE -> $name"
    bash "$REPO_ROOT/scripts/merge_checkpoint.sh" "$REPO_ROOT/$lora" "$name"
  fi
  exit 0
fi

echo ">>> sbatch $SLURM_DIR/${SLURM[$STAGE]}"
if [ "$DRY" = "1" ]; then exit 0; fi
out=$(sbatch "$SLURM_DIR/${SLURM[$STAGE]}" 2>&1)
echo "$out"
jid=$(echo "$out" | grep -oE '[0-9]+' | tail -1)
if [ -n "${MERGE[$STAGE]:-}" ] && [ -n "$jid" ]; then
  IFS='|' read -r lora name <<< "${MERGE[$STAGE]}"
  echo ">>> 训练后合并: merge.slurm -> $name (依赖 afterok:$jid)"
  sbatch --dependency=afterok:$jid "$SLURM_DIR/merge.slurm" "$REPO_ROOT/$lora" "$name"
fi