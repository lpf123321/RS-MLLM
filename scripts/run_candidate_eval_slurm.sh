#!/bin/bash
#SBATCH --job-name=rs_vlm_eval
#SBATCH --output=results/slurm_candidate_eval_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
PROJECT=${PROJECT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}

: "${CANDIDATE:?Submit with --export=ALL,CANDIDATE=<candidate-key>}"
DATASET="${DATASET:-vrsbench}"
TASK="${TASK:-vqa}"
STAGE="${STAGE:-fit}"
MAX_SAMPLES="${MAX_SAMPLES:-3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED=1
cd "${PROJECT}"
mkdir -p results

echo "=== CANDIDATE EVAL START $(date --iso-8601=seconds) ==="
echo "candidate=${CANDIDATE} dataset=${DATASET} task=${TASK} stage=${STAGE} max_samples=${MAX_SAMPLES}"
echo "python_bin=${PYTHON_BIN}"
"${PYTHON_BIN}" -u eval_candidates.py \
  --candidate "${CANDIDATE}" \
  --dataset "${DATASET}" \
  --task "${TASK}" \
  --stage "${STAGE}" \
  --max-samples "${MAX_SAMPLES}"
echo "=== CANDIDATE EVAL END $(date --iso-8601=seconds) ==="
