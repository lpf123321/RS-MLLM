#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ASSET_ROOT="${RS_MLLM_ASSET_ROOT:-${PACKAGE_ROOT}/.artifacts}"
UPSTREAM_ROOT="${RS_MLLM_UPSTREAM_ROOT:-${ASSET_ROOT}/upstream}"

# Defaults point only to repositories linked by the dataset authors and pin the
# revisions used while preparing this release.
MME_ID="${RS_MLLM_MME_ID:-yifanzhang114/MME-RealWorld}"
MME_REVISION="${RS_MLLM_MME_REVISION:-741cb8831ac86085bd54f678d13ca193e2334114}"
VRS_ID="${RS_MLLM_VRS_ID:-xiang709/VRSBench}"
VRS_REVISION="${RS_MLLM_VRS_REVISION:-6cee2968fd752a6d51c6cb2d18dded2bc0baa218}"
XLRS_LITE_ID="${RS_MLLM_XLRS_LITE_ID:-initiacms/XLRS-Bench-lite}"
XLRS_LITE_REVISION="${RS_MLLM_XLRS_LITE_REVISION:-e540ee2aa745ce9a83784ae76541ddb7f79f03ac}"
XLRS_GROUNDING_ID="${RS_MLLM_XLRS_GROUNDING_ID:-initiacms/XLRS-Bench_visual_grounding_en}"
XLRS_GROUNDING_REVISION="${RS_MLLM_XLRS_GROUNDING_REVISION:-2e01c12762aca045e2cf03c24b661c82a3e63209}"

command -v hf >/dev/null 2>&1 || {
  echo "hf is required: curl -LsSf https://hf.co/cli/install.sh | bash -s" >&2
  exit 2
}

mkdir -p "${UPSTREAM_ROOT}"

# MME's authors explicitly forbid third-party redistribution. Download only the
# remote-sensing archive from the repository linked by the authors. Running the
# download requires the reproducer to acknowledge the official terms.
if [[ ! -s "${UPSTREAM_ROOT}/mme/.download_complete" ]]; then
  [[ "${RS_MLLM_ACCEPT_MME_TERMS:-0}" == "1" ]] || {
    echo "Read the MME-RealWorld license and set RS_MLLM_ACCEPT_MME_TERMS=1 to acknowledge it." >&2
    exit 2
  }
  mkdir -p "${UPSTREAM_ROOT}/mme"
  hf download "${MME_ID}" --repo-type dataset --revision "${MME_REVISION}" \
    --include 'remote_sensing.tar.gz.part_*' \
    --local-dir "${UPSTREAM_ROOT}/mme"
  touch "${UPSTREAM_ROOT}/mme/.download_complete"
fi

if [[ ! -s "${UPSTREAM_ROOT}/vrsbench/.download_complete" ]]; then
  mkdir -p "${UPSTREAM_ROOT}/vrsbench"
  hf download "${VRS_ID}" Images_train.zip --repo-type dataset --revision "${VRS_REVISION}" \
    --local-dir "${UPSTREAM_ROOT}/vrsbench"
  touch "${UPSTREAM_ROOT}/vrsbench/.download_complete"
fi

if [[ ! -s "${UPSTREAM_ROOT}/xlrs_lite/.download_complete" ]]; then
  mkdir -p "${UPSTREAM_ROOT}/xlrs_lite"
  hf download "${XLRS_LITE_ID}" --repo-type dataset --revision "${XLRS_LITE_REVISION}" \
    --local-dir "${UPSTREAM_ROOT}/xlrs_lite"
  touch "${UPSTREAM_ROOT}/xlrs_lite/.download_complete"
fi

if [[ ! -s "${UPSTREAM_ROOT}/xlrs_grounding/.download_complete" ]]; then
  mkdir -p "${UPSTREAM_ROOT}/xlrs_grounding"
  hf download "${XLRS_GROUNDING_ID}" --repo-type dataset --revision "${XLRS_GROUNDING_REVISION}" \
    --include 'train/*' \
    --local-dir "${UPSTREAM_ROOT}/xlrs_grounding"
  touch "${UPSTREAM_ROOT}/xlrs_grounding/.download_complete"
fi

python "${SCRIPT_DIR}/extract_official_archives.py" \
  --mme-parts-dir "${UPSTREAM_ROOT}/mme" \
  --mme-output "${UPSTREAM_ROOT}/mme_extracted" \
  --vrs-zip "${UPSTREAM_ROOT}/vrsbench/Images_train.zip" \
  --vrs-output "${UPSTREAM_ROOT}/vrsbench_extracted"

python "${SCRIPT_DIR}/materialize_images.py" \
  --dataset-root "${ASSET_ROOT}/datasets" \
  --source "mme_realworld_rs=${UPSTREAM_ROOT}/mme_extracted" \
  --source "vrsbench=${UPSTREAM_ROOT}/vrsbench_extracted" \
  --source "xlrs=${UPSTREAM_ROOT}/xlrs_lite" \
  --source "xlrs_grounding=${UPSTREAM_ROOT}/xlrs_grounding"

echo "Official images materialized and hash-verified under ${ASSET_ROOT}/datasets/images."
