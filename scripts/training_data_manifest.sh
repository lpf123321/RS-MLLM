#!/bin/bash
# ============================================================
# 训练数据文件名映射（ModelScope 清晰名 <-> 训练脚本内部名）
#
# 被 scripts/package_training_data.sh 与 scripts/fetch_training_data.sh 共用，
# 保证打包与下载两侧改名一致。
#
# 用法：
#   source scripts/training_data_manifest.sh
#   for pair in "${DATA_FILES[@]}"; do
#       src=${pair%%::*}   # 源（发布清晰名）
#       dst=${pair##*::}   # 目标（训练脚本内部名）
#   done
#
# 命名约定：
#   发布名（ModelScope 上可见）用用途直白名；
#   内部名（训练脚本 DATA_PATH 引用）保持仓库既有约定。
# ============================================================

# 第一组：VRSbench 顶层文件（发布名 :: 内部名）
DATA_FILES=(
  "stage1_unified_sft_train.json::manifest_sft_train.json"
  "stage1_combined_train.json::combined_train.json"
  "grounding_domain_align_train.json::a1_domainalign.json"
  "change_anti_forget_train.json::a2_change_mix.json"
  "general_anti_forget_train.json::g_a2_mix.json"
  "caption_dual_domain_train.jsonl::expert_data_caption.jsonl"
)

# 第二组：expert_data/ 子目录文件
DATA_FILES_EXPERT=(
  "general_understanding_train.json::general_understanding.json"
  "referring_grounding_train.json::grounding.json"
  "change_detection_train.json::change.json"
  "stats.json::stats.json"
)

# 第三组：expert_data_v2/ 子目录文件
DATA_FILES_EXPERT_V2=(
  "general_understanding_train.json::general_understanding.json"
  "referring_grounding_train.json::grounding.json"
  "change_detection_train.json::change.json"
  "stats.json::stats.json"
)