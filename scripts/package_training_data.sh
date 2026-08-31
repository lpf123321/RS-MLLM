#!/bin/bash
# ============================================================
# 打包四专家训练所用清洗 json 数据 -> 供上传 ModelScope 的目录
#
# 输出: 一个目录（默认 transfer/rs_mllm_training_data），内含
#   finetune_framework/VRSbench/ 的 12 个 json/jsonl 清洗数据（相对结构）。
# 上传时把该目录内容直接推为 ModelScope dataset repo 即可。
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VRS="$REPO_ROOT/finetune_framework/VRSbench"
OUT="${1:-$REPO_ROOT/transfer/rs_mllm_training_data}"
D="$OUT/finetune_framework/VRSbench"

echo "打包到: $OUT"
mkdir -p "$D/expert_data" "$D/expert_data_v2"

# 顶层 json / jsonl（12 个中位于 VRSbench 直接层的）
for f in \
  manifest_sft_train.json \
  combined_train.json \
  a1_domainalign.json \
  a2_change_mix.json \
  g_a2_mix.json \
  expert_data_caption.jsonl \
; do
  [ -f "$VRS/$f" ] && cp "$VRS/$f" "$D/$f" && echo "  +$f" || echo "  !! 缺 $f"
done

# expert_data / expert_data_v2 子目录
for sub in expert_data expert_data_v2; do
  for f in "$VRS/$sub"/*.json; do
    cp "$f" "$D/$sub/$(basename "$f")" && echo "  +$sub/$(basename "$f")"
  done
done

echo "==== 打包完成 ===="
echo "上传: 把 $OUT 目录内容推至 ModelScope（例如 rs-mllm-training-data）"
echo "下载侧: 运行 bash scripts/fetch_training_data.sh"