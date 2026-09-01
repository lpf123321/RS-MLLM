#!/bin/bash
# 启动 vLLM Router: 4 专家实例(串行启动避免显存竞争) + 规则路由
# 用法: bash scripts/start_router.sh [--models-dir DIR] [--lora-dir DIR]
# 分卡: 卡0 = general(8001) + caption(8004); 卡1 = grounding(8002) + change(8003)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/evaluation/vllm_eval/.venv/bin/python"
MODELS="${1:-$HOME/router_models}"
LORAS="${2:-$HOME/lora_test}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$REPO_ROOT/evaluation/vllm_eval/.venv/bin:$PATH"

# expert:model:port:device:lora
EXPERTS=(
  "general:general_exp7_merged:8001:0:$LORAS/general"
  "grounding:ground_expert_update_merged:8002:1:$LORAS/grounding"
  "change:change_merged:8003:1:"
  "caption:caption_merged:8004:0:"
)

start_one() {
  local name="$1" model="$2" port="$3" dev="$4" lora="${5:-}"
  local cmd=("$PY" -m vllm.entrypoints.openai.api_server
    --model "$MODELS/$model" --served-model-name default --port "$port"
    --dtype bfloat16 --trust-remote-code
    --max-model-len 16384 --limit-mm-per-prompt '{"image": 2}'
    --gpu-memory-utilization 0.45)
  if [ -n "$lora" ]; then
    cmd+=(--enable-lora --max-lora-rank 32 --lora-modules "$name=$lora")
  fi
  echo "[router] 启动 $name (port $port, device $dev, model $model)"
  CUDA_VISIBLE_DEVICES="$dev" nohup "${cmd[@]}" > "$REPO_ROOT/logs/router_$name.log" 2>&1 &
  # 等该实例就绪(最长 8 分钟)
  for _ in $(seq 1 96); do
    if curl -s "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
      echo "[router] $name 就绪 (port $port)"
      return 0
    fi
    sleep 5
  done
  echo "[router] !! $name 启动超时, 查看 logs/router_$name.log" >&2
  return 1
}

mkdir -p "$REPO_ROOT/logs"
for spec in "${EXPERTS[@]}"; do
  IFS=: read -r name model port dev lora <<< "$spec"
  start_one "$name" "$model" "$port" "$dev" "$lora"
done
echo "[router] 全部就绪: general=8001 grounding=8002 change=8003 caption=8004"
echo "[router] 交互对话: python -m rsmllm.router --chat"
