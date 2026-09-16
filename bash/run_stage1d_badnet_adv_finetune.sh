#!/usr/bin/env bash

# Fine-tune a fresh BadNet0 copy on successful target-0 PGD endpoints from the
# existing Probe Top-500 selection.  The Model Zoo itself is never modified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-}"
MODEL_ZOO_SOURCE_ROOT="${MODEL_ZOO_SOURCE_ROOT:-${HOME}/backdoor-model-zoo}"
SELECTION_FILE="${SELECTION_FILE:-}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
BADNET_TRIGGER_PATH="${BADNET_TRIGGER_PATH:-${BACKDOORBENCH_ROOT}/resource/badnet/trigger_image.png}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_badnet_adv_finetune}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_ZOO_ROOT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar10" ]] || { echo "ERROR: CIFAR-10 directory is missing: ${DATA_ROOT}/cifar10" >&2; exit 1; }
[[ -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is missing: ${MODEL_ZOO_ROOT}" >&2; exit 1; }
[[ -f "${SELECTION_FILE}" ]] || { echo "ERROR: SELECTION_FILE must point to selected_probe_top500.csv" >&2; exit 1; }
[[ -f "${BADNET_TRIGGER_PATH}" ]] || { echo "ERROR: BadNet trigger is missing: ${BADNET_TRIGGER_PATH}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"
{
    echo "[$(date --iso-8601=seconds)] BadNet0 robust-sample adversarial fine-tuning"
    echo "[$(date --iso-8601=seconds)] selection=${SELECTION_FILE} model_zoo=${MODEL_ZOO_ROOT} device=cuda:0"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/badnet_adv_finetune.py" \
        --data-root "${DATA_ROOT}" \
        --model-zoo-root "${MODEL_ZOO_ROOT}" \
        --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}" \
        --selection-file "${SELECTION_FILE}" \
        --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
        --badnet-trigger-path "${BADNET_TRIGGER_PATH}" \
        --output-root "${OUTPUT_ROOT}" \
        --batch-size "${BATCH_SIZE}" \
        --eval-batch-size "${EVAL_BATCH_SIZE}" \
        --device cuda:0
} 2>&1 | tee -a "${LAUNCH_LOG}"

echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
