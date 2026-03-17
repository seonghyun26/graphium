#!/usr/bin/env bash
# Train GPS++ 800M from scratch on ADMET benchmark tasks (no pre-training).
# Uses the exact same model architecture as 01_finetune_admet.sh with gpspp_800M,
# so results are directly comparable (only difference: no pre-trained weights).
#
# Usage:
#   bash scripts/02_scratch_admet_800M.sh [gpu_id]
#
# Examples:
#   bash scripts/02_scratch_admet_800M.sh 4

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}
FINETUNE_DIM=${FINETUNE_DIM:-256}
ADDED_DEPTH=${ADDED_DEPTH:-4}

TAGS="['gpspp','scratch','admet']"

echo "=== Training GPS++ 800M from scratch on ADMET (baseline) ==="

for task in "${ADMET_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=gpspp_800M \
        accelerator=gpu \
        tasks=admet \
        $(wandb_flags "${TAGS}") \
        ++constants.raise_train_error=False \
        ++constants.task=${task} \
        ++datamodule.args.tdc_benchmark_names=${task} \
        ++datamodule.args.num_workers=0 \
        ++datamodule.args.featurization_n_jobs=0 \
        ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
        ++architecture.task_heads.${task}.depth=${ADDED_DEPTH} \
        ++trainer.model_checkpoint.save_last=False \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
