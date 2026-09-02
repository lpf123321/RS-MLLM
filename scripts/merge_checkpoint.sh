#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE= LORA= OUTPUT= DRY=0
while (($#)); do
  case "$1" in
    --base) BASE="${2:?--base requires a directory}"; shift 2;;
    --lora) LORA="${2:?--lora requires a directory}"; shift 2;;
    --output) OUTPUT="${2:?--output requires a directory}"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "Usage: merge_checkpoint.sh --base DIR --lora DIR --output DIR [--dry-run]" >&2; exit 2;;
  esac
done
[[ -n "$BASE" && -n "$LORA" && -n "$OUTPUT" ]] || {
  echo "--base, --lora and --output are required" >&2; exit 2;
}
if ((DRY == 1)); then printf '[merge dry-run] base=%s lora=%s output=%s\n' "$BASE" "$LORA" "$OUTPUT"; exit 0; fi
[[ -f "$BASE/config.json" ]] || { echo "invalid base model: $BASE" >&2; exit 2; }

lora_src="$LORA"
if [[ ! -f "$lora_src/adapter_config.json" ]]; then
  latest="$(find "$LORA" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -1)"
  [[ -n "$latest" && -f "$latest/adapter_config.json" ]] || { echo "invalid LoRA: $LORA" >&2; exit 2; }
  lora_src="$latest"
fi
[[ -n "$OUTPUT" ]] || { echo "--output is required" >&2; exit 2; }
if [[ -e "$OUTPUT" && -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite non-empty output: $OUTPUT" >&2; exit 2
fi
printf '[merge] base=%s lora=%s output=%s\n' "$BASE" "$lora_src" "$OUTPUT"
((DRY == 1)) && exit 0
source "$REPO_ROOT/scripts/training/env.sh"
activate_conda
export PYTHONPATH="$REPO_ROOT/training/distillation/expert_lora/src${PYTHONPATH:+:$PYTHONPATH}"
python -m expert_lora.compose \
  --base "$BASE" \
  --adapter "$lora_src" \
  --output "$OUTPUT"
[[ -f "$OUTPUT/config.json" ]] || { echo "merged output is not reloadable: $OUTPUT" >&2; exit 1; }
