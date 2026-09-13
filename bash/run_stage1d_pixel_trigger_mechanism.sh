#!/usr/bin/env bash

# CIFAR-10 pixel-space mechanism experiment.
# Each backdoor family independently selects 100 samples that satisfy:
# true label != 0, Clean0 targeted-PGD success, corresponding backdoor
# targeted-PGD success, and official-trigger prediction 0 on that backdoor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-}"
MODEL_ZOO_SOURCE_ROOT="${MODEL_ZOO_SOURCE_ROOT:-${HOME}/backdoor-model-zoo}"
TRIGGER_ARTIFACT_ROOT="${TRIGGER_ARTIFACT_ROOT:-}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_pixel_trigger_mechanism}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RANDOM_SEED="${RANDOM_SEED:-20260913}"

SSBA_ENCODER_PATH="${SSBA_ENCODER_PATH:-${DATA_ROOT}/stage1d_ssba_encoder/checkpoints/stage1d_cifar10_ssba_encoder.pth}"
SSBA_CONFIG_PATH="${SSBA_CONFIG_PATH:-${REPO_ROOT}/configs/stage1d_ssba_provenance.json}"
INPUTAWARE_STATE_PATH="${INPUTAWARE_STATE_PATH:-${TRIGGER_ARTIFACT_ROOT}/inputaware/seed0/netCGM.pt}"
ADAPTIVE_BLEND_TRIGGER_PATH="${ADAPTIVE_BLEND_TRIGGER_PATH:-${HOME}/backdoor-toolbox/triggers/hellokitty_32.png}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_ZOO_ROOT
export MODEL_ZOO_SOURCE_ROOT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar10" ]] || { echo "ERROR: CIFAR-10 directory missing: ${DATA_ROOT}/cifar10" >&2; exit 1; }
[[ -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is missing: ${MODEL_ZOO_ROOT}" >&2; exit 1; }
[[ -d "${TRIGGER_ARTIFACT_ROOT}" ]] || { echo "ERROR: TRIGGER_ARTIFACT_ROOT is missing: ${TRIGGER_ARTIFACT_ROOT}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"

ARGS=(
    --data-root "${DATA_ROOT}"
    --model-zoo-root "${MODEL_ZOO_ROOT}"
    --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}"
    --trigger-artifact-root "${TRIGGER_ARTIFACT_ROOT}"
    --backdoorbench-root "${BACKDOORBENCH_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --batch-size "${BATCH_SIZE}"
    --random-seed "${RANDOM_SEED}"
    --device cuda:0
    --inputaware-state-path "${INPUTAWARE_STATE_PATH}"
    --adaptive-blend-trigger-path "${ADAPTIVE_BLEND_TRIGGER_PATH}"
    --ssba-encoder-path "${SSBA_ENCODER_PATH}"
    --ssba-config-path "${SSBA_CONFIG_PATH}"
)

{
    echo "[$(date --iso-8601=seconds)] Stage 1D CIFAR-10 pixel trigger mechanism"
    echo "[$(date --iso-8601=seconds)] target=0 epsilon_attempts=1,1.5/255 cohort_size=100"
    echo "[$(date --iso-8601=seconds)] model_zoo=${MODEL_ZOO_ROOT} trigger_artifacts=${TRIGGER_ARTIFACT_ROOT}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/pixel_trigger_mechanism.py" "${ARGS[@]}"
} 2>&1 | tee -a "${LAUNCH_LOG}"

echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
