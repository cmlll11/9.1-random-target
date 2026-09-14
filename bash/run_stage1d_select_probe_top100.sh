#!/usr/bin/env bash

# One-time selection step for the layerwise mechanism experiments. This uses
# an existing Clean1--3-trained target-0 Probe to score Clean0's complete
# CIFAR-10 test split and writes the single selection file reused by WaNet and
# SSBA (and any other supported backdoor runner).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-}"
PROBE_ARCHIVE="${PROBE_ARCHIVE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_probe_top100_selection}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-100}"
RANDOM_SEED="${RANDOM_SEED:-20260914}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_ZOO_ROOT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar10" ]] || { echo "ERROR: CIFAR-10 directory is missing: ${DATA_ROOT}/cifar10" >&2; exit 1; }
[[ -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is missing: ${MODEL_ZOO_ROOT}" >&2; exit 1; }
[[ -f "${PROBE_ARCHIVE}" ]] || { echo "ERROR: existing Probe archive is missing: ${PROBE_ARCHIVE}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"
{
    echo "[$(date --iso-8601=seconds)] Selecting one shared Clean0 Probe Top-100"
    echo "[$(date --iso-8601=seconds)] probe_archive=${PROBE_ARCHIVE} model_zoo=${MODEL_ZOO_ROOT} device=cuda:0"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/select_probe_top100_once.py" \
        --data-root "${DATA_ROOT}" \
        --model-zoo-root "${MODEL_ZOO_ROOT}" \
        --probe-archive "${PROBE_ARCHIVE}" \
        --output-root "${OUTPUT_ROOT}" \
        --top-k 100 \
        --batch-size "${BATCH_SIZE}" \
        --random-seed "${RANDOM_SEED}" \
        --device cuda:0
} 2>&1 | tee -a "${LAUNCH_LOG}"
echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
