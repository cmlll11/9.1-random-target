#!/usr/bin/env bash

# Train the official Stage 1D-WT models on the complete CIFAR-10 training
# split.  The old hard-sample checkpoints are never used by this launcher.
# BackdoorBench handles the four qualified attack families used by this
# experiment: BadNet, Blended, WaNet and Input-Aware.  Adaptive-Blend is
# intentionally excluded because its official default run failed the ASR
# quality gate; its artifacts remain separate and are not used here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/mdl-uap/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/artifacts/models/stage1d_wt_official}"
BACKDOORBENCH_ROOT="${BACKDOORBENCH_ROOT:-${REPO_ROOT}/third_party/BackdoorBench}"
ADAPTIVE_BLEND_MODEL_PATH="${ADAPTIVE_BLEND_MODEL_PATH:-}"
ADAPTIVE_BLEND_TRIGGER_PATH="${ADAPTIVE_BLEND_TRIGGER_PATH:-}"
GPU_ID="${GPU_ID:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-1}"
RUN_TAG="${RUN_TAG:-stage1d_wt_$(date -u +%Y%m%dT%H%M%SZ)}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/cifar10" ]] || { echo "ERROR: full CIFAR-10 directory missing: ${DATA_ROOT}/cifar10" >&2; exit 1; }
[[ -d "${BACKDOORBENCH_ROOT}/attack" ]] || { echo "ERROR: BackdoorBench checkout missing: ${BACKDOORBENCH_ROOT}" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available()' || { echo "ERROR: CUDA unavailable" >&2; exit 1; }

mkdir -p "${MODEL_ROOT}/training_configs" "${MODEL_ROOT}/training_logs"
git -C "${BACKDOORBENCH_ROOT}" rev-parse HEAD > "${MODEL_ROOT}/training_configs/backdoorbench_commit.txt"
cp "${BACKDOORBENCH_ROOT}/config/attack/prototype/cifar10.yaml" "${MODEL_ROOT}/training_configs/clean_cifar10.yaml"
for config in badnet blended wanet inputaware; do
    cp "${BACKDOORBENCH_ROOT}/config/attack/${config}/default.yaml" "${MODEL_ROOT}/training_configs/${config}_default.yaml"
done

copy_result() {
    local group="$1" seed="$2" run="$3"
    local source="${BACKDOORBENCH_ROOT}/record/${run}/attack_result.pt"
    local destination="${MODEL_ROOT}/${group}/seed${seed}"
    mkdir -p "${destination}"

    if [[ -f "${source}" ]]; then
        # BackdoorBench attack scripts save the complete attack_result.pt.
        cp "${source}" "${destination}/attack_result.pt"
    elif [[ "${group}" == "clean_select_shared" && -f "${BACKDOORBENCH_ROOT}/record/${run}/clean_model.pth" ]]; then
        # BackdoorBench's prototype Clean attack intentionally saves only a
        # raw state_dict as clean_model.pth.  Wrap it in the same lightweight
        # payload expected by the experiment's model loader so that Clean and
        # backdoor checkpoints use one stable interface.  This fallback is
        # deliberately restricted to Clean; a missing backdoor attack_result
        # must remain a hard error.
        local clean_source="${BACKDOORBENCH_ROOT}/record/${run}/clean_model.pth"
        cp "${clean_source}" "${destination}/clean_model.pth"
        "${PYTHON_BIN}" - "${clean_source}" "${destination}/attack_result.pt" <<'PY'
import sys

import torch

source, destination = sys.argv[1:]
state = torch.load(source, map_location="cpu", weights_only=False)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
if not isinstance(state, dict):
    raise TypeError(f"Expected a state_dict in {source}, got {type(state).__name__}")
payload = {
    "model_name": "preactresnet18",
    "num_classes": 10,
    "model": state,
}
torch.save(payload, destination)
PY
    else
        echo "ERROR: official result missing: ${source}" >&2
        exit 1
    fi

    cp -f "${BACKDOORBENCH_ROOT}/record/${run}/info.pickle" "${destination}/" 2>/dev/null || true
    if [[ "${group}" == "wanet" ]]; then
        cp -f "${BACKDOORBENCH_ROOT}/record/${run}/identity_grid" "${destination}/state_identity_grid.pt" 2>/dev/null || true
        cp -f "${BACKDOORBENCH_ROOT}/record/${run}/noise_grid" "${destination}/state_noise_grid.pt" 2>/dev/null || true
        cp -f "${BACKDOORBENCH_ROOT}/record/${run}/state_dict.pt" "${destination}/state_dict.pt" 2>/dev/null || true
    fi
    if [[ "${group}" == "inputaware" ]]; then
        cp -f "${BACKDOORBENCH_ROOT}/record/${run}/netCGM.pt" "${destination}/netCGM.pt" 2>/dev/null || true
        cp -f "${BACKDOORBENCH_ROOT}/record/${run}/mask_state_dict.pt" "${destination}/mask_state_dict.pt" 2>/dev/null || true
    fi
}

train_clean() {
    local seed="$1" run="${RUN_TAG}_clean_seed${seed}"
    if [[ "${FORCE_RETRAIN}" == "0" && -f "${MODEL_ROOT}/clean_select_shared/seed${seed}/attack_result.pt" ]]; then return; fi
    (cd "${BACKDOORBENCH_ROOT}" && "${PYTHON_BIN}" attack/prototype.py \
        --yaml_path config/attack/prototype/cifar10.yaml --dataset_path "${DATA_ROOT}" \
        --save_folder_name "${run}" --random_seed "${seed}" --frequency_save 0 --device cuda:0) \
        2>&1 | tee "${MODEL_ROOT}/training_logs/clean_seed${seed}.log"
    copy_result clean_select_shared "${seed}" "${run}"
}

train_bdb() {
    local group="$1" script="$2" config="$3"
    local run="${RUN_TAG}_${group}_seed0"
    if [[ "${FORCE_RETRAIN}" == "0" && -f "${MODEL_ROOT}/${group}/seed0/attack_result.pt" ]]; then return; fi
    (cd "${BACKDOORBENCH_ROOT}" && "${PYTHON_BIN}" "${script}" \
        --yaml_path config/attack/prototype/cifar10.yaml --bd_yaml_path "${config}" \
        --dataset_path "${DATA_ROOT}" --save_folder_name "${run}" \
        --random_seed 0 --frequency_save 1 --device cuda:0) \
        2>&1 | tee "${MODEL_ROOT}/training_logs/${group}_seed0.log"
    copy_result "${group}" 0 "${run}"
}

for seed in 0 1 2 3; do train_clean "${seed}"; done
train_bdb badnet attack/badnet.py config/attack/badnet/default.yaml
train_bdb blended attack/blended.py config/attack/blended/default.yaml
train_bdb wanet attack/wanet.py config/attack/wanet/default.yaml
train_bdb inputaware attack/inputaware.py config/attack/inputaware/default.yaml

# Adaptive-Blend is not part of the strict quality gate.  If an official
# checkpoint was produced separately, copy it into the shared artifact tree
# so the mechanism analysis can include it as an exploratory group.
if [[ -n "${ADAPTIVE_BLEND_MODEL_PATH}" && -f "${ADAPTIVE_BLEND_MODEL_PATH}" ]]; then
    mkdir -p "${MODEL_ROOT}/adaptive_blend/seed0"
    cp "${ADAPTIVE_BLEND_MODEL_PATH}" "${MODEL_ROOT}/adaptive_blend/seed0/official_model.pt"
    if [[ -n "${ADAPTIVE_BLEND_TRIGGER_PATH}" && -f "${ADAPTIVE_BLEND_TRIGGER_PATH}" ]]; then
        cp "${ADAPTIVE_BLEND_TRIGGER_PATH}" "${MODEL_ROOT}/adaptive_blend/seed0/adaptive_blend_trigger.png"
    fi
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_stage1d_wt_models.py" \
    --data-root "${DATA_ROOT}" --model-root "${MODEL_ROOT}" --backdoorbench-root "${BACKDOORBENCH_ROOT}" \
    --output "${MODEL_ROOT}/stage1d_wt_model_gates.json" \
    --backdoor-groups "badnet,blended,wanet,inputaware,adaptive_blend" \
    --gate-exclude-groups "adaptive_blend"
echo "Official Stage 1D-WT model training complete: ${MODEL_ROOT}"
