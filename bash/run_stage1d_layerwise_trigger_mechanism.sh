#!/usr/bin/env bash

# Reuse one previously trained Probe and one shared Clean0 Top-100 selection,
# then run the Clean0-vs-backdoor layerwise mechanism analysis on exactly those
# samples. Probe fitting and Clean0 selection are separate from this runner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-}"
MODEL_ZOO_SOURCE_ROOT="${MODEL_ZOO_SOURCE_ROOT:-${HOME}/backdoor-model-zoo}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
TRIGGER_ARTIFACT_ROOT="${TRIGGER_ARTIFACT_ROOT:-}"
BACKDOOR_ALIAS="${BACKDOOR_ALIAS:-badnet0}"
BADNET_TRIGGER_PATH="${BADNET_TRIGGER_PATH:-${BACKDOORBENCH_ROOT}/resource/badnet/trigger_image.png}"
WANET_STATE_PATH="${WANET_STATE_PATH:-}"
SSBA_ENCODER_PATH="${SSBA_ENCODER_PATH:-${DATA_ROOT}/stage1d_ssba_encoder/checkpoints/stage1d_cifar10_ssba_encoder.pth}"
SSBA_CONFIG_PATH="${SSBA_CONFIG_PATH:-${REPO_ROOT}/configs/stage1d_ssba_provenance.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_layerwise_probe_top100_${BACKDOOR_ALIAS}}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-100}"
RANDOM_SEED="${RANDOM_SEED:-20260914}"
SELECTION_FILE="${SELECTION_FILE:-}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_ZOO_ROOT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar10" ]] || { echo "ERROR: CIFAR-10 directory is missing: ${DATA_ROOT}/cifar10" >&2; exit 1; }
[[ -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is missing: ${MODEL_ZOO_ROOT}" >&2; exit 1; }
[[ -f "${SELECTION_FILE}" ]] || {
    echo "ERROR: SELECTION_FILE must point to the shared selected_probe_top100.csv" >&2
    echo "       Run bash/run_stage1d_select_probe_top100.sh once using the existing Probe archive." >&2
    exit 1
}
case "${BACKDOOR_ALIAS}" in
    badnet0|blended0|wanet0|inputaware0|ssba0) ;;
    *) echo "ERROR: BACKDOOR_ALIAS must be badnet0, blended0, wanet0, inputaware0, or ssba0" >&2; exit 1 ;;
esac
if [[ "${BACKDOOR_ALIAS}" == "badnet0" ]]; then
    [[ -f "${BADNET_TRIGGER_PATH}" ]] || { echo "ERROR: BadNet trigger is missing: ${BADNET_TRIGGER_PATH}" >&2; exit 1; }
fi
if [[ "${BACKDOOR_ALIAS}" != "badnet0" ]]; then
    [[ -d "${TRIGGER_ARTIFACT_ROOT}" ]] || { echo "ERROR: TRIGGER_ARTIFACT_ROOT is missing: ${TRIGGER_ARTIFACT_ROOT}" >&2; exit 1; }
fi
if [[ "${BACKDOOR_ALIAS}" == "ssba0" ]]; then
    [[ -f "${SSBA_ENCODER_PATH}" ]] || { echo "ERROR: SSBA encoder is missing: ${SSBA_ENCODER_PATH}" >&2; exit 1; }
    [[ -f "${SSBA_CONFIG_PATH}" ]] || { echo "ERROR: SSBA provenance config is missing: ${SSBA_CONFIG_PATH}" >&2; exit 1; }
fi

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"
if [[ "${BACKDOOR_ALIAS}" != "badnet0" ]]; then
    PREFLIGHT_PATH="${OUTPUT_ROOT}/preflight_${BACKDOOR_ALIAS}.json"
    PREFLIGHT_ARGS=(
        --model-zoo-root "${MODEL_ZOO_ROOT}"
        --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}"
        --backdoorbench-root "${BACKDOORBENCH_ROOT}"
        --trigger-artifact-root "${TRIGGER_ARTIFACT_ROOT}"
        --backdoor-alias "${BACKDOOR_ALIAS}"
        --wanet-state-path "${WANET_STATE_PATH}"
        --ssba-encoder-path "${SSBA_ENCODER_PATH}"
        --ssba-config-path "${SSBA_CONFIG_PATH}"
        --device cuda:0
        --output "${PREFLIGHT_PATH}"
    )
    echo "[$(date --iso-8601=seconds)] Running ${BACKDOOR_ALIAS} Model Zoo/trigger preflight" | tee -a "${LAUNCH_LOG}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_layerwise_backdoor_prerequisites.py" "${PREFLIGHT_ARGS[@]}" 2>&1 | tee -a "${LAUNCH_LOG}"
fi

ARGS=(
    --data-root "${DATA_ROOT}"
    --model-zoo-root "${MODEL_ZOO_ROOT}"
    --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}"
    --backdoorbench-root "${BACKDOORBENCH_ROOT}"
    --trigger-artifact-root "${TRIGGER_ARTIFACT_ROOT:-${BACKDOORBENCH_ROOT}}"
    --backdoor-alias "${BACKDOOR_ALIAS}"
    --badnet-trigger-path "${BADNET_TRIGGER_PATH}"
    --wanet-state-path "${WANET_STATE_PATH}"
    --ssba-encoder-path "${SSBA_ENCODER_PATH}"
    --ssba-config-path "${SSBA_CONFIG_PATH}"
    --output-root "${OUTPUT_ROOT}"
    --selection-file "${SELECTION_FILE}"
    --top-k 100
    --batch-size "${BATCH_SIZE}"
    --random-seed "${RANDOM_SEED}"
    --device cuda:0
)

{
    echo "[$(date --iso-8601=seconds)] Stage 1D shared Probe Top-100 layerwise Clean0-vs-${BACKDOOR_ALIAS} mechanism analysis"
    echo "[$(date --iso-8601=seconds)] Reusing one Probe selection -> Clean0/${BACKDOOR_ALIAS}"
    echo "[$(date --iso-8601=seconds)] model_zoo=${MODEL_ZOO_ROOT} device=cuda:0 batch_size=${BATCH_SIZE}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/layerwise_probe_top100_trigger_mechanism.py" "${ARGS[@]}"
} 2>&1 | tee -a "${LAUNCH_LOG}"

echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
