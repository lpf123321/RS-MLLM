#!/bin/bash
#SBATCH --job-name=quant950
#SBATCH --output=${RSMLLM_LOGS_DIR:-REPO/logs}/quant950_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
# 量化模型 950 清单评测（与 bf16 基线同口径）
set -euo pipefail
PROJECT=${RSMLLM_REPO:-REPO}
ROOT=REPO0260727
ENV=${ROOT}/environments/vllm0
UV=${HOME}/.local/bin/uv
cd "${PROJECT}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=${PROJECT}/scripts:${PROJECT}/scripts/specprobe_fix${PYTHONPATH:+:$PYTHONPATH}

MODEL="${1:?用法: quant_eval_950.slurm <model_dir> <profile> <outdir>}"
PROFILE="${2:?profile: mmerestore_bf16|mmerestore_w8a8|mmerestore_gptq}"
OUTDIR="${3:?outdir}"

"${UV}" run --project "${ENV}" --offline python -m rs_mllm_dataset_fixes.vision_opd_vllm_eval \
  --manifest ${RSMLLM_TILED:-REPO}/tiled_rag/e0_pair/e2_manifest.jsonl \
  --model "$MODEL" --model-profile "$PROFILE" --output-dir "$OUTDIR" \
  --batch-size 64 --max-num-seqs 64 --max-model-len 16384

# 评分（clean_correct 官方口径）
"${UV}" run --project "${ENV}" --offline python - << 'PYEOF'
import json, sys
sys.path.insert(0, "REPOaset-fixes/src")
from rs_mllm_dataset_fixes.schema import Sample
from rs_mllm_dataset_fixes.scoring import score_prediction
outdir, = sys.argv[1:]
corr = tot = 0
for line in open(f"{outdir}/prediction_attempts.jsonl"):
    r = json.loads(line)
    s = Sample.from_dict(r["sample"])
    sc = score_prediction(s, r.get("prediction", ""))
    corr += bool(sc.get("clean_correct")) if isinstance(sc, dict) else bool(sc)
    tot += 1
print(f"accuracy: {corr}/{tot} = {corr/tot:.4f}")
PYEOF
echo "=== quant950 done ==="
