#!/usr/bin/env bash

# Train the Clean1--3 Probe, select Clean0 CIFAR-10 Top-100, then run the
# Clean0-vs-BadNet0 layerwise mechanism analysis on exactly those samples.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-}"
MODEL_ZOO_SOURCE_ROOT="${MODEL_ZOO_SOURCE_ROOT:-${HOME}/backdoor-model-zoo}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
BADNET_TRIGGER_PATH="${BADNET_TRIGGER_PATH:-${BACKDOORBENCH_ROOT}/resource/badnet/trigger_image.png}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_layerwise_trigger_mechanism}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-100}"
RANDOM_SEED="${RANDOM_SEED:-20260914}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_ZOO_ROOT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar10" ]] || { echo "ERROR: CIFAR-10 directory is missing: ${DATA_ROOT}/cifar10" >&2; exit 1; }
[[ -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is missing: ${MODEL_ZOO_ROOT}" >&2; exit 1; }
[[ -f "${BADNET_TRIGGER_PATH}" ]] || { echo "ERROR: BadNet trigger is missing: ${BADNET_TRIGGER_PATH}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"

ARGS=(
    --data-root "${DATA_ROOT}"
    --model-zoo-root "${MODEL_ZOO_ROOT}"
    --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}"
    --backdoorbench-root "${BACKDOORBENCH_ROOT}"
    --badnet-trigger-path "${BADNET_TRIGGER_PATH}"
    --output-root "${OUTPUT_ROOT}"
    --top-k 100
    --batch-size "${BATCH_SIZE}"
    --random-seed "${RANDOM_SEED}"
    --device cuda:0
)

{
    echo "[$(date --iso-8601=seconds)] Stage 1D Probe Top-100 layerwise Clean0-vs-BadNet0 mechanism analysis"
    echo "[$(date --iso-8601=seconds)] Clean1-3 Probe -> Clean0 Top-100 -> Clean0/BadNet0"
    echo "[$(date --iso-8601=seconds)] model_zoo=${MODEL_ZOO_ROOT} device=cuda:0 batch_size=${BATCH_SIZE}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/layerwise_probe_top100_trigger_mechanism.py" "${ARGS[@]}"
} 2>&1 | tee -a "${LAUNCH_LOG}"

echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
