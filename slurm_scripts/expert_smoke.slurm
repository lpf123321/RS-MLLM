#!/bin/bash
#SBATCH --job-name=expert-smoke
#SBATCH --output=${RSMLLM_LOGS_DIR:-REPO/logs}/expert_smoke_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# 4 个专家合并模型 vLLM 加载 smoke: 加载 + 短生成 + 显存/TTFT
set -euo pipefail
ROOT=REPO0260727
UV=${HOME}/.local/bin/uv
BASE=${RSMLLM_MODELS_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/models}/lora_expert/_merge
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0

cat > /tmp/smoke_experts.py << 'PYEOF'
import json, sys, time
from vllm import LLM, SamplingParams

models = {
    "general": f"{sys.argv[1]}/general_exp7_merged",
    "ground": f"{sys.argv[1]}/ground_expert_update_merged",
    "change": f"{sys.argv[1]}/change_merged",
    "caption": f"{sys.argv[1]}/caption_merged",
}
prompt = [{"type": "text", "text": "Describe the image briefly."}]
# 无图纯文本 probe: 用 VL 模型文本输入
for name, path in models.items():
    t0 = time.time()
    try:
        llm = LLM(model=path, tensor_parallel_size=1, gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True)
        t_load = time.time() - t0
        sp = SamplingParams(temperature=0, max_tokens=32)
        outs = llm.generate([{"prompt": "Hello, introduce yourself in one sentence."}], sp)
        out_txt = outs[0].outputs[0].text[:80]
        print(f"SMOKE {name}: LOAD={t_load:.1f}s OK gen={out_txt!r}")
        del llm
        import gc; gc.collect()
    except Exception as e:
        print(f"SMOKE {name}: FAIL {type(e).__name__}: {str(e)[:200]}")
print("ALL_SMOKE_DONE")
PYEOF
"${UV}" run --project "${ROOT}/environments/vllm0" --offline python /tmp/smoke_experts.py "${BASE}"
echo "=== expert smoke done ==="
