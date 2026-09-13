#!/usr/bin/env bash

# Measure successful target-PGD feature-direction concentration on the fixed
# Clean0 Probe Top-100.  This runner intentionally does not use trigger
# success to filter the cohort.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-}"
MODEL_ZOO_SOURCE_ROOT="${MODEL_ZOO_SOURCE_ROOT:-${HOME}/backdoor-model-zoo}"
SELECTION_FILE="${SELECTION_FILE:?set SELECTION_FILE to selected_targeted_robust_samples.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_target0_adv_concentration}"
EPSILON_PIXELS="${EPSILON_PIXELS:-0.75,1,1.25}"
MODEL_ALIASES="${MODEL_ALIASES:-clean0,badnet0,blended0,wanet0,inputaware0,ssba0,adaptive_blend01}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_ZOO_ROOT
export MODEL_ZOO_SOURCE_ROOT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar100" ]] || { echo "ERROR: CIFAR-100 directory missing: ${DATA_ROOT}/cifar100" >&2; exit 1; }
[[ -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is missing: ${MODEL_ZOO_ROOT}" >&2; exit 1; }
[[ -f "${SELECTION_FILE}" ]] || { echo "ERROR: selection file is missing: ${SELECTION_FILE}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"
{
    echo "[$(date --iso-8601=seconds)] Stage 1D target=0 successful-PGD concentration"
    echo "[$(date --iso-8601=seconds)] epsilon=${EPSILON_PIXELS}/255 aliases=${MODEL_ALIASES}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/robust_attack_concentration.py" \
        --data-root "${DATA_ROOT}" \
        --model-zoo-root "${MODEL_ZOO_ROOT}" \
        --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}" \
        --selected-samples "${SELECTION_FILE}" \
        --output-root "${OUTPUT_ROOT}" \
        --model-aliases "${MODEL_ALIASES}" \
        --epsilon-pixels "${EPSILON_PIXELS}" \
        --batch-size "${BATCH_SIZE}" \
        --device cuda:0
} 2>&1 | tee -a "${LAUNCH_LOG}"
echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
