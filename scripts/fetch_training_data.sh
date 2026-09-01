#!/bin/bash
# ============================================================
# 下载训练数据集（清洗后的 json/jsonl）—— 懒加载: ModelScope 首次下载, 本地命中复用
#
# 通过 rsmllm.data 完成:
#   1) get_dataset("training")   -> 下载到 <仓库>/datasets (默认 RSMLLM_DATASETS_CACHE)
#   2) restore_training_data()   -> 发布名还原内部名, 放到 finetune_framework/VRSbench/
# 训练脚本(DATA_PATH)直接读内部名, 无需额外步骤。
#
# 用法:
#   bash scripts/fetch_training_data.sh
#   环境变量: MODELSCOPE_REPO(默认 yasumi/rs-mllm-datasets), RSMLLM_DATASETS_CACHE,
#             MODELSCOPE_API_TOKEN, MODELSCOPE_OFFLINE=1(纯离线)
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/finetune_framework/VRSbench"

# 若本地已有数据则跳过
if [ -f "$DEST/manifest_sft_train.json" ] && [ -f "$DEST/expert_data_caption.jsonl" ]; then
  echo "[fetch_training_data] 训练数据已存在，跳过下载。"
  exit 0
fi

echo "[fetch_training_data] 从 ModelScope 懒加载训练数据 ..."

# 用评测环境 python(含 modelscope + 登录凭据); 否则系统 python
EVAL_PY="$REPO_ROOT/evaluation/vllm_eval/.venv/bin/python"
MS_PY="$EVAL_PY"
if ! "$MS_PY" -c "import modelscope" 2>/dev/null; then
  MS_PY="python"
fi
if ! "$MS_PY" -c "import modelscope" 2>/dev/null; then
  echo "[fetch_training_data] 未找到含 modelscope 的 python，请先 pip install modelscope。"
  exit 1
fi

PYTHONPATH="$REPO_ROOT:$PYTHONPATH" "$MS_PY" - <<'PY'
import os
from rsmllm.data import get_dataset, restore_training_data

repo = os.environ.get("MODELSCOPE_REPO", "yasumi/rs-mllm-datasets")
cache = os.environ.get("RSMLLM_DATASETS_CACHE")
root = get_dataset(repo, cache_dir=cache)
print(f"[fetch_training_data] 数据集缓存: {root}")
dest = restore_training_data(dataset_root=root)
print(f"[fetch_training_data] 已还原训练json到: {dest}")
PY

echo "[fetch_training_data] 完成。训练数据位于 $DEST"