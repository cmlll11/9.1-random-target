#!/usr/bin/env bash

# Reuse the completed pixel-space BadNet cohort and endpoints.  This runner
# performs forward hooks only; it does not run PGD, Probe fitting, or sample
# selection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-}"
MODEL_ZOO_SOURCE_ROOT="${MODEL_ZOO_SOURCE_ROOT:-${HOME}/backdoor-model-zoo}"
SOURCE_ENDPOINT_ARRAYS="${SOURCE_ENDPOINT_ARRAYS:-}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_layerwise_trigger_mechanism}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-100}"
RANDOM_SEED="${RANDOM_SEED:-20260914}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_ZOO_ROOT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is missing: ${MODEL_ZOO_ROOT}" >&2; exit 1; }
[[ -f "${SOURCE_ENDPOINT_ARRAYS}" ]] || { echo "ERROR: endpoint archive is missing: ${SOURCE_ENDPOINT_ARRAYS}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"

ARGS=(
    --model-zoo-root "${MODEL_ZOO_ROOT}"
    --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}"
    --source-endpoint-arrays "${SOURCE_ENDPOINT_ARRAYS}"
    --output-root "${OUTPUT_ROOT}"
    --batch-size "${BATCH_SIZE}"
    --random-seed "${RANDOM_SEED}"
    --device cuda:0
)
if [[ -n "${SOURCE_RUN_DIR}" ]]; then
    ARGS+=(--source-run-dir "${SOURCE_RUN_DIR}")
fi

{
    echo "[$(date --iso-8601=seconds)] Stage 1D layerwise Clean0-vs-BadNet0 mechanism analysis"
    echo "[$(date --iso-8601=seconds)] source_endpoint_arrays=${SOURCE_ENDPOINT_ARRAYS}"
    echo "[$(date --iso-8601=seconds)] model_zoo=${MODEL_ZOO_ROOT} device=cuda:0 batch_size=${BATCH_SIZE}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/layerwise_trigger_mechanism.py" "${ARGS[@]}"
} 2>&1 | tee -a "${LAUNCH_LOG}"

echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
