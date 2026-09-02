#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${1:-}"
[[ "$KIND" = general || "$KIND" = grounding ]] || {
  echo "Usage: bash scripts/evaluate_training35.sh general|grounding --model DIR --tag NAME [--subsets DIR] [--batch-size N]" >&2
  exit 2
}
shift
MODEL=
TAG=
SUBSETS="$REPO_ROOT/outputs/training35/eval_subsets"
BATCH_SIZE=4
while (($#)); do
  case "$1" in
    --model) MODEL="${2:?--model requires a directory}"; shift 2 ;;
    --tag) TAG="${2:?--tag requires a name}"; shift 2 ;;
    --subsets) SUBSETS="${2:?--subsets requires a directory}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?--batch-size requires an integer}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -f "$MODEL/config.json" ]] || { echo "invalid model: $MODEL" >&2; exit 2; }
[[ -n "$TAG" ]] || { echo "--tag is required" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "--batch-size must be positive" >&2; exit 2; }

OUTPUT_ROOT="$REPO_ROOT/outputs/training35/trend/$KIND/$TAG"
RESULTS="$OUTPUT_ROOT/results.json"
PREDICTIONS="$OUTPUT_ROOT/predictions.json"
[[ ! -e "$RESULTS" ]] || { echo "refusing to overwrite: $RESULTS" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"

# shellcheck source=training/env.sh
source "$REPO_ROOT/scripts/training/env.sh"
activate_conda
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$KIND" = general ]]; then
  MME="$SUBSETS/mme_1000.jsonl"
  XLRS="$SUBSETS/xlrs_1000.jsonl"
  VRS="$SUBSETS/vrsbench_vqa_1000.jsonl"
  [[ -f "$MME" && -f "$XLRS" && -f "$VRS" ]] || {
    echo "missing General fixed subsets under $SUBSETS" >&2
    exit 2
  }
  OVERRIDES="$(printf '{"mme":"%s","xlrs":"%s","vrsbench":"%s"}' "$MME" "$XLRS" "$VRS")"
  DATASETS=(mme xlrs vrsbench)
  SUBTASK=vqa
else
  VRS="$SUBSETS/vrsbench_referring_1000.jsonl"
  XLRS="$SUBSETS/xlrs_grounding_1000.jsonl"
  [[ -f "$VRS" && -f "$XLRS" ]] || {
    echo "missing Grounding fixed subsets under $SUBSETS" >&2
    exit 2
  }
  OVERRIDES="$(printf '{"vrsbench":"%s","xlrs_grounding":"%s"}' "$VRS" "$XLRS")"
  DATASETS=(vrsbench xlrs_grounding)
  SUBTASK=referring
fi

python "$REPO_ROOT/scripts/evaluate_training35_entry.py" \
  --adapter qwen35vl \
  --model_path "$MODEL" \
  --datasets "${DATASETS[@]}" \
  --subtask "$SUBTASK" \
  --data_path_overrides "$OVERRIDES" \
  --eval_batch_size "$BATCH_SIZE" \
  --output "$RESULTS" \
  --save_predictions "$PREDICTIONS"

[[ -f "$RESULTS" ]] || { echo "evaluator returned without creating results: $RESULTS" >&2; exit 1; }
echo "[training35-eval] results=$RESULTS"
