#!/bin/bash
# Evaluate the corrected DivPrune implementation on fixed 1,000-sample subsets.
# Submit with: sbatch evaluation/run_divprune_1000.slurm
# Optional overrides: VRS_MODEL, LEVIR_MODEL, DATA_ROOT, RESULTS_DIR,
# SAMPLE_COUNT, SAMPLE_SEED, KEEP_RATIOS, OVERWRITE.
# Example: KEEP_RATIOS="0.50" sbatch evaluation/run_divprune_1000.slurm
#SBATCH --job-name=divprune_1000
#SBATCH --output=logs/divprune_1000_%j.out
#SBATCH --error=logs/divprune_1000_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -o pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VRS_MODEL="${VRS_MODEL:-$REPO_ROOT/finetune_framework/VRSbench_qwen35vl/output/stage2_high_merged}"
LEVIR_MODEL="${LEVIR_MODEL:-$REPO_ROOT/finetune_framework/LEVIR-CC/output/stage2_high_merged}"
DATA_ROOT="${DATA_ROOT:-${RSMLLM_DATA_ROOT:-$(cd $(dirname $(readlink -f "${BASH_SOURCE[0]:-$0}"))/.. && pwd)/datasets}}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/evaluation/results/divprune_1000}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-2026}"
KEEP_RATIOS="${KEEP_RATIOS:-0.50 0.25}"
CONDA_ENV="${CONDA_ENV:-rs_mllm}"

source /opt/app/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RESULTS_DIR" "$REPO_ROOT/logs"

for required_path in "$VRS_MODEL" "$LEVIR_MODEL"; do
    if [[ ! -f "$required_path/config.json" ]]; then
        echo "Missing merged model: $required_path" >&2
        exit 2
    fi
done

failures=0
for ratio in $KEEP_RATIOS; do
    case "$ratio" in
        0.5|0.50) label="r50" ;;
        0.25) label="r25" ;;
        *)
            echo "Unsupported keep ratio: $ratio" >&2
            exit 2
            ;;
    esac

    for dataset in mme xlrs vrsbench levircc; do
        model="$VRS_MODEL"
        family="vrs"
        if [[ "$dataset" == "levircc" ]]; then
            model="$LEVIR_MODEL"
            family="levir"
        fi

        output="$RESULTS_DIR/results_${dataset}_${family}_divprune_${label}_${SAMPLE_COUNT}.json"
        if [[ -s "$output" && "${OVERWRITE:-0}" != "1" ]]; then
            echo "[SKIP] $output"
            continue
        fi

        echo "[START] dataset=$dataset ratio=$ratio model=$model"
        if python -m evaluation.main \
            --model_path "$model" \
            --adapter qwen35_divprune \
            --datasets "$dataset" \
            --random_samples "$SAMPLE_COUNT" \
            --sample_seed "$SAMPLE_SEED" \
            --eval_batch_size 1 \
            --keep_ratio "$ratio" \
            --pruner_seed "$SAMPLE_SEED" \
            --output "$output"; then
            echo "[DONE] $output"
        else
            echo "[FAIL] dataset=$dataset ratio=$ratio" >&2
            failures=$((failures + 1))
        fi
    done
done

exit "$failures"
