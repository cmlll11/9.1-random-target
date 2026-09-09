#!/usr/bin/env bash

# Run Stage 1D-WT after official model retraining and quality gating.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/artifacts/models/stage1d_wt_official}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
QUALITY_REPORT="${QUALITY_REPORT:-${MODEL_ROOT}/stage1d_wt_model_gates.json}"
ADAPTIVE_BLEND_ROOT="${ADAPTIVE_BLEND_ROOT:-}"
INPUTAWARE_STATE_PATH="${INPUTAWARE_STATE_PATH:-${MODEL_ROOT}/inputaware/seed0/netCGM.pt}"
ADAPTIVE_BLEND_TRIGGER_PATH="${ADAPTIVE_BLEND_TRIGGER_PATH:-${MODEL_ROOT}/adaptive_blend/seed0/adaptive_blend_trigger.png}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/stage1d_wrong_target_trigger_alignment}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar100" ]] || { echo "ERROR: CIFAR-100 directory missing: ${DATA_ROOT}/cifar100" >&2; exit 1; }
[[ -f "${QUALITY_REPORT}" ]] || { echo "ERROR: quality report missing: ${QUALITY_REPORT}" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available()' || { echo "ERROR: CUDA unavailable" >&2; exit 1; }

for seed in 1 2 3; do
    [[ -f "${MODEL_ROOT}/clean_select_shared/seed${seed}/attack_result.pt" ]] || { echo "ERROR: Clean reference checkpoint missing for seed${seed}" >&2; exit 1; }
done
[[ -f "${MODEL_ROOT}/clean_select_shared/seed0/attack_result.pt" ]] || { echo "ERROR: Clean test checkpoint missing" >&2; exit 1; }
for group in badnet blended wanet inputaware adaptive_blend; do
    if [[ "${group}" == "adaptive_blend" ]]; then
        [[ -f "${MODEL_ROOT}/adaptive_blend/seed0/official_model.pt" ]] || { echo "ERROR: Adaptive-Blend checkpoint missing" >&2; exit 1; }
    else
        [[ -f "${MODEL_ROOT}/${group}/seed0/attack_result.pt" ]] || { echo "ERROR: ${group} checkpoint missing" >&2; exit 1; }
    fi
done

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
LAUNCH_LOG="${OUTPUT_ROOT}/launch_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "${OUTPUT_ROOT}"
ARGS=(
    --data-root "${DATA_ROOT}"
    --model-root "${MODEL_ROOT}"
    --backdoorbench-root "${BACKDOORBENCH_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --quality-report "${QUALITY_REPORT}"
    --adaptive-blend-root "${ADAPTIVE_BLEND_ROOT}"
    --inputaware-state-path "${INPUTAWARE_STATE_PATH}"
    --adaptive-blend-trigger-path "${ADAPTIVE_BLEND_TRIGGER_PATH}"
    --batch-size "${BATCH_SIZE}"
    --device cuda:0
)
{
    echo "[$(date --iso-8601=seconds)] Stage 1D-WT official-trigger alignment"
    echo "[$(date --iso-8601=seconds)] targets=1,3,7 epsilon=1,1.5/255 groups=badnet,blended,wanet,inputaware,adaptive_blend"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/trigger_alignment_wrong_target.py" "${ARGS[@]}"
} 2>&1 | tee -a "${LAUNCH_LOG}"
echo "[$(date --iso-8601=seconds)] launch log: ${LAUNCH_LOG}" | tee -a "${LAUNCH_LOG}"
