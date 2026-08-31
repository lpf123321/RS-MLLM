#!/bin/bash
# Fast preflight for the XLRS memory benchmark. It does not load a model.
# Usage: sbatch --wait evaluation/smoke_xlrs_memory.slurm
# Override DATA_PATH/CONDA_ENV when the cluster data layout differs.
#SBATCH --job-name=xlrs_mem_smoke
#SBATCH --output=evaluation/logs/xlrs_mem_smoke_%j.out
#SBATCH --error=evaluation/logs/xlrs_mem_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT_VALUE="${DATA_ROOT:-${RSMLLM_DATA_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/datasets}}"
DATA_PATH="${DATA_PATH:-$DATA_ROOT_VALUE/XLRS-Bench-lite/xlrs.jsonl}"
CONDA_ENV="${CONDA_ENV:-rs_mllm}"

source /opt/app/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u
cd "$REPO_ROOT"

python3 - "$DATA_PATH" <<'PY'
import sys
from pathlib import Path

from evaluation.evalsets import xlrs

path = Path(sys.argv[1])
samples = xlrs.load_data(str(path))
assert len(samples) >= 1000, f"XLRS has only {len(samples)} samples: {path}"
print(f"XLRS preflight OK: {len(samples)} samples, using first 1000 deterministically")
PY

python3 -m evaluation.benchmark_memory --help >/dev/null
echo "benchmark_memory CLI preflight OK"
