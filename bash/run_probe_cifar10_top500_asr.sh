#!/usr/bin/env bash

# Train a target-0 Probe on Clean1--3 CIFAR-10 train samples, select one
# shared Clean0 CIFAR-10 test Top-500, and compare targeted-PGD ASR across
# Clean0 and the registered seed-0 backdoor models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-}"
MODEL_ZOO_SOURCE_ROOT="${MODEL_ZOO_SOURCE_ROOT:-${HOME}/backdoor-model-zoo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1e_cifar10_probe_top500_asr}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_COUNT="${TRAIN_COUNT:-1000}"
TOP_K="${TOP_K:-500}"
RANDOM_SEED="${RANDOM_SEED:-20260914}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_ZOO_ROOT
export MODEL_ZOO_SOURCE_ROOT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar10" ]] || { echo "ERROR: CIFAR-10 directory missing: ${DATA_ROOT}/cifar10" >&2; exit 1; }
[[ -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is missing: ${MODEL_ZOO_ROOT}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"
{
    echo "[$(date --iso-8601=seconds)] CIFAR-10 Clean1-3 Probe -> Clean0 Top-${TOP_K} PGD ASR"
    echo "[$(date --iso-8601=seconds)] target=0 eps=1,1.5/255 train_count=${TRAIN_COUNT} device=cuda:0"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/probe_cifar10_top500_asr.py" \
        --data-root "${DATA_ROOT}" \
        --model-zoo-root "${MODEL_ZOO_ROOT}" \
        --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --target 0 \
        --top-k "${TOP_K}" \
        --train-count "${TRAIN_COUNT}" \
        --analysis-eps-pixels "1,1.5" \
        --batch-size "${BATCH_SIZE}" \
        --random-seed "${RANDOM_SEED}" \
        --device cuda:0
} 2>&1 | tee -a "${LAUNCH_LOG}"
echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
