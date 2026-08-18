#!/bin/bash
# Measure peak CUDA memory on the fixed 1,000-sample XLRS subset in one SLURM job.
# Submit one job only: sbatch evaluation/run_xlrs_memory.slurm
# Optional overrides: MODEL_PATH, DATA_PATH, RESULTS_DIR, CONDA_ENV,
#                     SAMPLE_SEED, PRUNER_SEED, COMPILE_MODEL.
# Set OVERWRITE=1 to rerun configurations whose JSON already exists.
#SBATCH --job-name=xlrs_mem_1000
#SBATCH --output=evaluation/logs/xlrs_mem_1000_%j.out
#SBATCH --error=evaluation/logs/xlrs_mem_1000_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -o pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/home/u2025310307/models/Qwen3.5-4B}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/evaluation/data/xlrs_memory_1000.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/evaluation/results/xlrs_memory_1000}"
CONDA_ENV="${CONDA_ENV:-rs_mllm}"
SAMPLE_SEED="${SAMPLE_SEED:-2026}"
PRUNER_SEED="${PRUNER_SEED:-2026}"
COMPILE_ARGS=()
if [[ "${COMPILE_MODEL:-0}" == "1" ]]; then
    COMPILE_ARGS+=(--compile_model)
fi

mkdir -p "$RESULTS_DIR" "$REPO_ROOT/evaluation/logs"
source /opt/app/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u
cd "$REPO_ROOT"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Data file not found: $DATA_PATH" >&2
    exit 2
fi

declare -a METHODS=(baseline uniform random mmtok l2norm scope divprune fourier)
failures=0
started_at=$(date +%s)

write_summary() {
    python3 - "$RESULTS_DIR" "$MODEL_PATH" "$DATA_PATH" "$SAMPLE_SEED" <<'PY'
import json
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
summary = []
for path in sorted(results_dir.glob("xlrs_memory_*.json")):
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    summary.append({
        "method": item.get("method"),
        "keep_ratio": item.get("keep_ratio"),
        "torch_peak_allocated_mib": item.get("torch_peak_allocated_mib"),
        "torch_peak_reserved_mib": item.get("torch_peak_reserved_mib"),
        "nvidia_smi_process_peak_mib": item.get("nvidia_smi_process_peak_mib"),
        "samples": item.get("samples"),
        "result_file": str(path),
    })
summary.sort(key=lambda x: (x["torch_peak_allocated_mib"] is None, x["torch_peak_allocated_mib"] or 0))
output = results_dir / "xlrs_memory_summary.json"
output.write_text(json.dumps({
    "dataset": "xlrs",
    "samples": 1000,
    "sample_seed": int(sys.argv[4]),
    "model_path": sys.argv[2],
    "data_path": sys.argv[3],
    "results": summary,
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[SUMMARY] wrote {len(summary)} results to {output}", flush=True)
PY
}

for method in "${METHODS[@]}"; do
    if [[ "$method" == baseline ]]; then
        ratios=("1.0")
    else
        ratios=("0.5" "0.25")
    fi

    for keep_ratio in "${ratios[@]}"; do
        if [[ "$method" == baseline ]]; then
            label=baseline
        elif [[ "$keep_ratio" == "0.5" ]]; then
            label="${method}_r50"
        else
            label="${method}_r25"
        fi
        output="$RESULTS_DIR/xlrs_memory_${label}.json"

        if [[ -s "$output" && "${OVERWRITE:-0}" != "1" ]]; then
            echo "[SKIP] $label already exists: $output"
            continue
        fi

        echo "[START] method=$method keep_ratio=$keep_ratio output=$output"
        if python3 -m evaluation.benchmark_memory \
            --model_path "$MODEL_PATH" \
            --dataset xlrs \
            --data_path "$DATA_PATH" \
            --method "$method" \
            --keep_ratio "$keep_ratio" \
            --sample_count 1000 \
            --sample_seed "$SAMPLE_SEED" \
            --pruner_seed "$PRUNER_SEED" \
            --output "$output" \
            "${COMPILE_ARGS[@]}"; then
            echo "[DONE] $label"
            # Refresh the aggregate immediately so partial jobs remain useful.
            write_summary
        else
            echo "[FAIL] $label" >&2
            failures=$((failures + 1))
        fi
    done
done

# Also create an empty/complete summary when every configuration was skipped.
write_summary

elapsed=$(( $(date +%s) - started_at ))
echo "Completed in ${elapsed}s; failures=${failures}; results=${RESULTS_DIR}"
[[ "$failures" -eq 0 ]]
