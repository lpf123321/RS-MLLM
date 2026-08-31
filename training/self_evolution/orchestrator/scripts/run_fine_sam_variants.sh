#!/usr/bin/env bash
set -euo pipefail

STRICT_UNIT="cvsearch-recovered-full-r0-eval2.service"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${CONDA_BIN:-conda}
CONFIGS=(
  "${PROJECT_ROOT}/configs/vopd_qwen3vl_2b_fine_sam_trace_one_fifth.yaml"
  "${PROJECT_ROOT}/configs/vopd_qwen3vl_2b_fine_sam_replace_one_fifth.yaml"
)

# Do not contend with the strict experiment on GPUs 0-3. The queue is a user
# service, so it survives terminal and Codex session exits.
while systemctl --user is-active --quiet "${STRICT_UNIT}"; do
  sleep 60
done

cd "${PROJECT_ROOT}"
for config_path in "${CONFIGS[@]}"; do
  "${PYTHON_BIN}" run -n cvsearch python -m evolve --config "${config_path}" prepare
  "${PYTHON_BIN}" run -n cvsearch python -m evolve --config "${config_path}" smoke --round 0
  "${PYTHON_BIN}" run -n cvsearch python -m evolve --config "${config_path}" run-loop --stage full --max-rounds 1
done
