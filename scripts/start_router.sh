#!/bin/bash
# 启动 vLLM Router: 4 专家实例(串行启动避免显存竞争) + 规则路由
# 用法: bash scripts/start_router.sh [--models-dir DIR] [--lora-dir DIR]
# 分卡: 卡0 = general(8001) + caption(8004); 卡1 = grounding(8002) + change(8003)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="$REPO_ROOT/evaluation/vllm_eval/.venv/bin/python"
MODELS="$HOME/router_models"
LORAS="$HOME/lora_test"
MODELS_SET=0
LORAS_SET=0
while (($#)); do
  case "$1" in
    --models-dir)
      MODELS="${2:?--models-dir requires a directory}"
      MODELS_SET=1
      shift 2
      ;;
    --lora-dir)
      LORAS="${2:?--lora-dir requires a directory}"
      LORAS_SET=1
      shift 2
      ;;
    --help|-h)
      printf 'Usage: bash scripts/start_router.sh [--models-dir DIR] [--lora-dir DIR]\n'
      exit 0
      ;;
    *)
      if ((MODELS_SET == 0)); then
        MODELS="$1"
        MODELS_SET=1
      elif ((LORAS_SET == 0)); then
        LORAS="$1"
        LORAS_SET=1
      else
        echo "unknown argument: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done
resolve_dir() {
  local path="${1/#\~\//$HOME/}"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$REPO_ROOT/$path"
  fi
}
MODELS="$(resolve_dir "$MODELS")"
LORAS="$(resolve_dir "$LORAS")"
[[ -x "$PY" ]] || { echo "missing evaluator Python: $PY" >&2; exit 2; }
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
  local model_path="$MODELS/$model"
  [[ -f "$model_path/config.json" ]] || {
    echo "missing model config: $model_path/config.json" >&2
    return 2
  }
  if [[ -n "$lora" && ! -d "$lora" ]]; then
    echo "missing LoRA directory: $lora" >&2
    return 2
  fi
  local cmd=("$PY" -m vllm.entrypoints.openai.api_server
    --model "$model_path" --served-model-name default --port "$port"
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
