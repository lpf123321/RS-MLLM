#!/usr/bin/env bash
# Standalone Transformers token-pruning entry.
#
# This path is deliberately separate from rsmllm.sh's vLLM route: the pruning
# adapters patch Qwen's Transformers forward and are not part of the vLLM
# evaluator. All arguments are forwarded to the existing, resumable Delta
# pruning evaluator.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    printf 'missing root Transformers environment: %s\n' "${PY}" >&2
    printf 'run bash setup.sh first\n' >&2
    exit 2
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${PY}" -m evaluation.run_delta_prune_eval "$@"
