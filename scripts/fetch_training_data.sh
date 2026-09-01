#!/bin/bash
# ============================================================
# 下载训练数据集（清洗后的 json/jsonl）—— 从 ModelScope 拉取
#
# 说明:
#   - 四专家训练所需的 12 个清洗 json 数据已打包上传至 ModelScope。
#   - 下载后自动放置到 finetune_framework/VRSbench/ 对应位置，
#     与训练脚本（scripts/training/*.slurm）的 DATA_PATH 一致。
#   - 图片（assets/<hash>.png）不在此包内，见 scripts/fetch_raw_images.sh
#
# 用法:
#   bash scripts/fetch_training_data.sh                       # 用默认 repo
#   MODELSCOPE_REPO=<repo_id> bash scripts/fetch_training_data.sh
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/finetune_framework/VRSbench"
TMP="$REPO_ROOT/.data_download"

# TODO: 上传后填写实际 ModelScope 仓库 id，例如 "xxx/RS-MLLM-training-data"
MODELSCOPE_REPO="${MODELSCOPE_REPO:-REPLACE_WITH_YOUR_MODELSCOPE_REPO_ID}"

# 若本地已有数据则跳过
if [ -f "$DEST/manifest_sft_train.json" ] && [ -f "$DEST/expert_data_caption.jsonl" ]; then
  echo "[fetch_training_data] 训练数据已存在，跳过下载。"
  exit 0
fi

echo "[fetch_training_data] 从 ModelScope 下载训练数据: $MODELSCOPE_REPO"

# 优先使用 modelscope SDK；否则尝试 git clone（需安装 git-lfs）
if python -c "import modelscope" 2>/dev/null; then
  mkdir -p "$TMP"
  python - "$MODELSCOPE_REPO" "$TMP" <<'PY'
import sys, shutil, os
from modelscope.hub.api import HubApi
repo_id, dst = sys.argv[1], sys.argv[2]
api = HubApi()
api.snapshot_download(repo_id, cache_dir=dst)
PY
  SRC="$TMP"
else
  SRC="$TMP"
  git clone --depth 1 "https://www.modelscope.cn/${MODELSCOPE_REPO}.git" "$TMP" || \
    { echo "git clone 失败，请先 pip install modelscope"; exit 1; }
fi

# 将下载的文件还原为训练脚本内部名（发布名 -> 内部名，见 training_data_manifest.sh）
source "$REPO_ROOT/scripts/training_data_manifest.sh"
mkdir -p "$DEST"

# 顶层文件
for pair in "${DATA_FILES[@]}"; do
  pub="${pair%%::*}"; int="${pair##*::}"
  src="$SRC/$pub"
  [ -f "$src" ] || src="$SRC/finetune_framework/VRSbench/$pub"
  if [ -f "$src" ]; then
    cp -n "$src" "$DEST/$int" && echo "  +$int   (<- $pub)"
  else
    echo "  !! 未找到 $pub（跳过）"
  fi
done

# expert_data / expert_data_v2 子目录
declare -A SUBS=([expert_data]="DATA_FILES_EXPERT" [expert_data_v2]="DATA_FILES_EXPERT_V2")
for sub in expert_data expert_data_v2; do
  mkdir -p "$DEST/$sub"
  var="${SUBS[$sub]}"; eval "list=(\"\${$var[@]}\")"
  for pair in "${list[@]}"; do
    pub="${pair%%::*}"; int="${pair##*::}"
    src="$SRC/$sub/$pub"; [ -f "$src" ] || src="$SRC/finetune_framework/VRSbench/$sub/$pub"
    if [ -f "$src" ]; then
      cp -n "$src" "$DEST/$sub/$int" && echo "  +$sub/$int   (<- $pub)"
    else
      echo "  !! 未找到 $sub/$pub（跳过）"
    fi
  done
done

echo "[fetch_training_data] 完成。数据位于 $DEST"
ls -la "$DEST" | head