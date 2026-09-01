#!/bin/bash
# ============================================================
# 打包四专家训练所用清洗 json 数据 -> 供上传 ModelScope 的目录
#
# 打包后的文件用"用途直白名"（发布名），供上传 ModelScope；
# 训练脚本内部仍引用原内部名，由 fetch_training_data.sh 下载时做反向映射。
# 两侧映射见 scripts/training_data_manifest.sh。
#
# 用法:
#   bash scripts/package_training_data.sh [输出目录]
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VRS="$REPO_ROOT/finetune_framework/VRSbench"
OUT="${1:-$REPO_ROOT/transfer/rs_mllm_training_data}"
D="$OUT/finetune_framework/VRSbench"

echo "打包到: $OUT（发布名 / ModelScope 上传用）"
mkdir -p "$D/expert_data" "$D/expert_data_v2"

source "$REPO_ROOT/scripts/training_data_manifest.sh"

# 顶层文件
for pair in "${DATA_FILES[@]}"; do
  pub="${pair%%::*}"; int="${pair##*::}"
  if [ -f "$VRS/$int" ]; then
    cp "$VRS/$int" "$D/$pub" && echo "  +$pub   (=$int)"
  else
    echo "  !! 缺 $int"
  fi
done

# expert_data / expert_data_v2
declare -A SUBS=([expert_data]="DATA_FILES_EXPERT" [expert_data_v2]="DATA_FILES_EXPERT_V2")
for sub in expert_data expert_data_v2; do
  [[ -d "$VRS/$sub" ]] || { echo "  !! 缺 $sub/"; continue; }
  var="${SUBS[$sub]}"; eval "list=(\"\${$var[@]}\")"
  for pair in "${list[@]}"; do
    pub="${pair%%::*}"; int="${pair##*::}"
    if [ -f "$VRS/$sub/$int" ]; then
      cp "$VRS/$sub/$int" "$D/$sub/$pub" && echo "  +$sub/$pub   (=$int)"
    else
      echo "  !! 缺 $sub/$int"
    fi
  done
done

echo "==== 打包完成（发布名）===="
echo "上传: 把 $OUT 目录内容推至 ModelScope"
echo "下载侧: bash scripts/fetch_training_data.sh 会自动还原为训练脚本内部名"