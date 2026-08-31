#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_ROOT=${PIPELINE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PIPELINE_ROOT}/artifacts}
PYTHON_BIN=${TRAJECTORY_PYTHON:-python}
GENERATOR=${PIPELINE_ROOT}/offpolicy/generate_teacher_trajectories.py
PREPARER=${PIPELINE_ROOT}/offpolicy/prepare_offpolicy_data.py
VALIDATOR=${PIPELINE_ROOT}/offpolicy/validate_offpolicy_data.py
TEACHER_PATH=${TEACHER_MODEL:-${ARTIFACT_ROOT}/sft/full-vision-opd-9b-sft-merged}
TRAJECTORY_DIR=${ARTIFACT_ROOT}/offpolicy/trajectories/full_sft_vision_opd_9b
LOG_DIR=${PIPELINE_ROOT}/logs/offpolicy/teacher_trajectory_generation
OUTPUT_PARQUET=${ARTIFACT_ROOT}/offpolicy/data/formal_teacher_trajectories_18413.parquet
AUDIT_PATH=${PIPELINE_ROOT}/reports/offpolicy_formal_data_audit.json

mkdir -p "${TRAJECTORY_DIR}" "${LOG_DIR}"

run_shard() {
    local gpu=$1
    local dataset=$2
    local shard_index=$3
    local num_shards=$4
    local input=${PIPELINE_ROOT}/data/train/${dataset}.jsonl
    local suffix
    suffix=$(printf 'part-%05d-of-%05d' "${shard_index}" "${num_shards}")
    local output=${TRAJECTORY_DIR}/${dataset}.${suffix}.jsonl
    local log=${LOG_DIR}/${dataset}.${suffix}.log

    echo "[$(date --iso-8601=seconds)] START gpu=${gpu} dataset=${dataset} shard=${shard_index}/${num_shards}" \
        | tee "${log}"
    (
        cd "${PIPELINE_ROOT}" || exit 1
        CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONPATH="${PIPELINE_ROOT}:${PYTHONPATH:-}" \
        "${PYTHON_BIN}" "${GENERATOR}" \
            --dataset "${dataset}" \
            --input "${input}" \
            --model-path "${TEACHER_PATH}" \
            --output "${output}" \
            --shard-index "${shard_index}" \
            --num-shards "${num_shards}" \
            --batch-size 8 \
            --max-new-tokens 64 \
            --temperature 1.0 \
            --top-p 1.0 \
            --seed 20260809 \
            --resume
    ) >>"${log}" 2>&1
    local status=$?
    echo "[$(date --iso-8601=seconds)] END status=${status} dataset=${dataset} shard=${shard_index}/${num_shards}" \
        | tee -a "${log}"
    return "${status}"
}

pids=()
names=()
for entry in "0:1:5" "1:2:5" "2:3:5" "3:4:5" "4:0:10" "5:1:10"; do
    gpu=${entry%%:*}
    rest=${entry#*:}
    shard=${rest%%:*}
    num_shards=${rest#*:}
    run_shard "${gpu}" vrsbench "${shard}" "${num_shards}" &
    pids+=("$!")
    names+=("vrsbench-${shard}-of-${num_shards}")
done
run_shard 6 levircc 0 1 &
pids+=("$!"); names+=("levircc-0")

# MME and XLRS are short, so they share GPU 7 sequentially.
(
    run_shard 7 mme 0 1
    run_shard 7 xlrs 0 1
) &
pids+=("$!"); names+=("mme-xlrs")

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "PASS ${names[$index]}"
    else
        echo "FAIL ${names[$index]}" >&2
        failed=1
    fi
done
if (( failed != 0 )); then
    echo "Teacher trajectory generation failed; parquet preparation was not run." >&2
    exit 1
fi

PYTHONPATH="${PIPELINE_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${PREPARER}" \
    --trajectory-dir "${TRAJECTORY_DIR}" \
    --output "${OUTPUT_PARQUET}"

"${PYTHON_BIN}" "${VALIDATOR}" "${OUTPUT_PARQUET}" \
    --expected-records 18413 \
    --output "${AUDIT_PATH}"

echo "Teacher trajectories and formal off-policy parquet are ready: ${OUTPUT_PARQUET}"
