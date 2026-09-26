#!/usr/bin/env bash

# Pixel-space removal of each sample's paired Clean0 adversarial direction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${HOME}/8.11/data}"
MODEL_ZOO_ROOT="${MODEL_ZOO_ROOT:-${HOME}/model_zoo}"
MODEL_ZOO_SOURCE_ROOT="${MODEL_ZOO_SOURCE_ROOT:-${HOME}/backdoor-model-zoo}"
SELECTION_FILE="${SELECTION_FILE:-}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/results}"
TRIGGER_ARTIFACT_ROOT="${TRIGGER_ARTIFACT_ROOT:-${HOME}/8.11/artifacts/models/stage1d_wt_official}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_adversarial_residual_decoupling}"
GPU_ID="${GPU_ID:-auto}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-20}"
MAX_GPU_MEMORY_MB="${MAX_GPU_MEMORY_MB:-2048}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RANDOM_SEED="${RANDOM_SEED:-20260926}"

SSBA_ENCODER_PATH="${SSBA_ENCODER_PATH:-${DATA_ROOT}/stage1d_ssba_encoder/checkpoints/stage1d_cifar10_ssba_encoder.pth}"
SSBA_CONFIG_PATH="${SSBA_CONFIG_PATH:-${REPO_ROOT}/configs/stage1d_ssba_provenance.json}"
INPUTAWARE_STATE_PATH="${INPUTAWARE_STATE_PATH:-${TRIGGER_ARTIFACT_ROOT}/inputaware/seed0/netCGM.pt}"
ADAPTIVE_BLEND_TRIGGER_PATH="${ADAPTIVE_BLEND_TRIGGER_PATH:-${HOME}/backdoor-toolbox/triggers/hellokitty_32.png}"

export MODEL_ZOO_ROOT
export MODEL_ZOO_SOURCE_ROOT
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar10" ]] || { echo "ERROR: CIFAR-10 directory missing: ${DATA_ROOT}/cifar10" >&2; exit 1; }
[[ -d "${MODEL_ZOO_ROOT}" ]] || { echo "ERROR: MODEL_ZOO_ROOT is missing: ${MODEL_ZOO_ROOT}" >&2; exit 1; }
[[ -f "${SELECTION_FILE}" ]] || { echo "ERROR: set SELECTION_FILE to the existing shared selected_probe_top100.csv" >&2; exit 1; }
[[ -d "${TRIGGER_ARTIFACT_ROOT}" ]] || { echo "ERROR: TRIGGER_ARTIFACT_ROOT is missing: ${TRIGGER_ARTIFACT_ROOT}" >&2; exit 1; }
[[ -d "${BACKDOORBENCH_ROOT}/resource" ]] || { echo "ERROR: BackdoorBench resources missing: ${BACKDOORBENCH_ROOT}" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi is required to check GPU occupancy before training." >&2; exit 1; }
[[ "${MAX_GPU_UTILIZATION}" =~ ^[0-9]+$ && "${MAX_GPU_MEMORY_MB}" =~ ^[0-9]+$ ]] || { echo "ERROR: GPU occupancy limits must be non-negative integers." >&2; exit 1; }

GPU_STATS="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
[[ -n "${GPU_STATS}" ]] || { echo "ERROR: could not read GPU occupancy from nvidia-smi." >&2; exit 1; }
if [[ "${GPU_ID}" == "auto" ]]; then
    GPU_CANDIDATE="$(printf '%s\n' "${GPU_STATS}" | awk -F, -v max_mem="${MAX_GPU_MEMORY_MB}" -v max_util="${MAX_GPU_UTILIZATION}" '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3); if ($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $2 <= max_mem && $3 <= max_util) print $3, $2, $1}' | sort -k1,1n -k2,2n | awk 'NR == 1 {first=$0} END {print first}')"
    [[ -n "${GPU_CANDIDATE}" ]] || { echo "ERROR: no idle GPU found (limits: utilization<=${MAX_GPU_UTILIZATION}%, memory<=${MAX_GPU_MEMORY_MB} MiB). Wait for an idle GPU; do not force a busy one." >&2; exit 1; }
    read -r _GPU_UTIL _GPU_MEMORY GPU_ID <<< "${GPU_CANDIDATE}"
else
    [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || { echo "ERROR: GPU_ID must be 'auto' or a numeric GPU index." >&2; exit 1; }
    GPU_ROW="$(printf '%s\n' "${GPU_STATS}" | awk -F, -v wanted="${GPU_ID}" '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3); if ($1 == wanted && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/) print $3, $2, $1}')"
    [[ -n "${GPU_ROW}" ]] || { echo "ERROR: GPU ${GPU_ID} is not visible to nvidia-smi." >&2; exit 1; }
    read -r _GPU_UTIL _GPU_MEMORY _GPU_INDEX <<< "${GPU_ROW}"
    if (( _GPU_UTIL > MAX_GPU_UTILIZATION || _GPU_MEMORY > MAX_GPU_MEMORY_MB )); then
        echo "ERROR: GPU ${GPU_ID} is busy (${_GPU_UTIL}% utilization, ${_GPU_MEMORY} MiB used); wait or choose an idle GPU." >&2
        exit 1
    fi
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${OUTPUT_ROOT}"
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_${LAUNCH_ID}.log"
PID_FILE="${OUTPUT_ROOT}/launch_${LAUNCH_ID}.pid"
printf '%s\n' "$$" > "${PID_FILE}"
ARGS=(
    --data-root "${DATA_ROOT}"
    --model-zoo-root "${MODEL_ZOO_ROOT}"
    --model-zoo-source-root "${MODEL_ZOO_SOURCE_ROOT}"
    --selection-file "${SELECTION_FILE}"
    --cache-root "${CACHE_ROOT}"
    --trigger-artifact-root "${TRIGGER_ARTIFACT_ROOT}"
    --backdoorbench-root "${BACKDOORBENCH_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --epsilon-pixels 1,1.5
    --steps 100
    --restarts 3
    --batch-size "${BATCH_SIZE}"
    --random-seed "${RANDOM_SEED}"
    --device cuda:0
    --inputaware-state-path "${INPUTAWARE_STATE_PATH}"
    --adaptive-blend-trigger-path "${ADAPTIVE_BLEND_TRIGGER_PATH}"
    --ssba-encoder-path "${SSBA_ENCODER_PATH}"
    --ssba-config-path "${SSBA_CONFIG_PATH}"
)

{
    echo "[$(date --iso-8601=seconds)] Stage 1D pixel adversarial-residual decoupling"
    echo "[$(date --iso-8601=seconds)] target=0 eps=1,1.5/255 steps=100 restarts=3 cohort=existing Top-100"
    echo "[$(date --iso-8601=seconds)] physical_gpu=${GPU_ID} utilization=${_GPU_UTIL}% memory_used=${_GPU_MEMORY}MiB (limits ${MAX_GPU_UTILIZATION}%/${MAX_GPU_MEMORY_MB}MiB)"
    echo "[$(date --iso-8601=seconds)] selection=${SELECTION_FILE} model_zoo=${MODEL_ZOO_ROOT} cache=${CACHE_ROOT}"
    echo "[$(date --iso-8601=seconds)] launch_id=${LAUNCH_ID} pid=$$ pid_file=${PID_FILE}"
    printf '[%s] command:' "$(date --iso-8601=seconds)"
    printf ' %q' "${PYTHON_BIN}" "${REPO_ROOT}/scripts/adversarial_residual_decoupling.py" "${ARGS[@]}"
    printf '\n'
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/adversarial_residual_decoupling.py" "${ARGS[@]}"
} 2>&1 | tee -a "${LAUNCH_LOG}"

echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
