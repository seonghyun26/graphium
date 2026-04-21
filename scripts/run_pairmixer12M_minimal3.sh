#!/usr/bin/env bash
# Minimal 3-run PairMixer 12M screening on 5 focus ADMET tasks.
#
# Runs three fine-tuning schedules from the latest toymix_dti_esmc_v2
# PairMixer 12M checkpoint on one task per ADMET category:
#   1. full unfreeze at epoch 50
#   2. unfreeze top 4 layers, then full unfreeze at epoch 50
#   3. unfreeze top 4 layers, then full unfreeze at epoch 10
#
# Usage:
#   bash scripts/run_pairmixer12M_minimal3.sh [gpu_id] [checkpoint]

set -euo pipefail

cd "$(dirname "$0")/.."

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate graphium 2>/dev/null || true
fi
export PATH="/home/shpark/.conda/envs/graphium/bin:$PATH"

source "$(dirname "$0")/common.sh"

MODEL=pairmixer_12M
DEVICE=${1:-5}
CKPT=${2:-}
FINETUNE_DIM=${FINETUNE_DIM:-256}
ADDED_DEPTH=${ADDED_DEPTH:-4}

if [[ -z "${CKPT}" ]]; then
    CKPT=$(find models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M \
        -name '*epochepoch=099*.ckpt' | sort | tail -n1)
fi
if [[ -z "${CKPT}" ]]; then
    CKPT=$(find models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M \
        -name '*.ckpt' | sort | tail -n1)
fi
if [[ -z "${CKPT}" ]]; then
    echo "Error: no PairMixer 12M toymix_dti_esmc_v2 checkpoint found."
    exit 1
fi

FOCUS_TASKS=(
    caco2_wang
    ppbr_az
    cyp2d6_veith
    clearance_hepatocyte_az
    ld50_zhu
)

RUNS=(
    "rep5_u0_e50|0|50"
    "rep5_u4_e50|4|50"
    "rep5_u4_e10|4|10"
)

LOG_DIR="logs/pairmixer12M_minimal3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=== PairMixer 12M minimal 3-run screen ===" | tee "${LOG_DIR}/summary.txt"
echo "GPU: ${DEVICE}" | tee -a "${LOG_DIR}/summary.txt"
echo "Checkpoint: ${CKPT}" | tee -a "${LOG_DIR}/summary.txt"
echo "Tasks: ${FOCUS_TASKS[*]}" | tee -a "${LOG_DIR}/summary.txt"
echo "Log dir: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.txt"

for spec in "${RUNS[@]}"; do
    IFS='|' read -r label unfreeze_depth epoch_unfreeze_all <<< "${spec}"
    log_file="${LOG_DIR}/${label}.log"

    echo "" | tee -a "${LOG_DIR}/summary.txt"
    echo ">>> ${label} | start $(date '+%F %T')" | tee -a "${LOG_DIR}/summary.txt"
    echo "    unfreeze_depth=${unfreeze_depth} epoch_unfreeze_all=${epoch_unfreeze_all}" \
        | tee -a "${LOG_DIR}/summary.txt"

    {
        for task in "${FOCUS_TASKS[@]}"; do
            echo "--- ${label} | ${task} ---"

            CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
                model=${MODEL} \
                accelerator=gpu \
                tasks=admet \
                $(wandb_flags "['${MODEL}','finetune','admet','toymix_dti_esmc_v2','${label}','rep5']") \
                ++constants.task=${task} \
                ++constants.name=finetuning_${task}_${MODEL}_${label} \
                +finetuning=admet_toymix_dti \
                ++finetuning.task=${task} \
                ++datamodule.args.tdc_benchmark_names=${task} \
                ++finetuning.pretrained_model="'${CKPT}'" \
                ++finetuning.unfreeze_pretrained_depth=${unfreeze_depth} \
                ++finetuning.epoch_unfreeze_all=${epoch_unfreeze_all} \
                ++finetuning.finetuning_head.in_dim=${FINETUNE_DIM} \
                ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
                ++finetuning.new_out_dim=${FINETUNE_DIM} \
                ++finetuning.finetuning_head.depth=${ADDED_DEPTH} \
                ++finetuning.added_depth=${ADDED_DEPTH} \
                ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
                ++trainer.model_checkpoint.dirpath=models_checkpoints/admet-minimal3/toymix_dti_esmc_v2/${MODEL}/${label}/${task}/ \
            || echo "WARN: ${label} ${task} failed, continuing..."

            sleep 1
        done
    } 2>&1 | tee "${log_file}"

    echo ">>> ${label} | done $(date '+%F %T')" | tee -a "${LOG_DIR}/summary.txt"
done

echo "" | tee -a "${LOG_DIR}/summary.txt"
echo "All three runs finished at $(date '+%F %T')" | tee -a "${LOG_DIR}/summary.txt"
