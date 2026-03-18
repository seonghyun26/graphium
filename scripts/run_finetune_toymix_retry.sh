#!/usr/bin/env bash
# Retry failed ADMET finetuning tasks for the ToyMix-pretrained GPS++ 800M.
#
# Usage:
#   bash scripts/run_finetune_toymix_retry.sh [gpu_id]

set -euo pipefail
source "$(dirname "$0")/common.sh"

DEVICE=${1:-4}
CKPT=/home/shpark/prj-molrepr/graphium/models_checkpoints/small-dataset/gpspp_800M/2026-03-17_21-26-42_20260317_212642/last.ckpt
FINETUNE_DIM=256
ADDED_DEPTH=4
FINETUNING_CONFIG=admet

FAILED_TASKS=(caco2_wang bioavailability_ma lipophilicity_astrazeneca solubility_aqsoldb)
TAGS="['gpspp','finetune','admet','toymix','gpspp_800M']"

echo "=== Retrying ${#FAILED_TASKS[@]} failed tasks (toymix pretrained, GPU ${DEVICE}) ==="

for task in "${FAILED_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=gpspp \
        accelerator=gpu \
        tasks=admet \
        $(wandb_flags "${TAGS}") \
        ++constants.raise_train_error=False \
        ++constants.task=${task} \
        ++finetuning.task=${task} \
        ++datamodule.args.tdc_benchmark_names=${task} \
        ++datamodule.args.num_workers=0 \
        ++datamodule.args.featurization_n_jobs=0 \
        +finetuning=${FINETUNING_CONFIG} \
        ++finetuning.pretrained_model=${CKPT} \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=none \
        ++finetuning.finetuning_head.in_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
        ++finetuning.new_out_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.depth=${ADDED_DEPTH} \
        ++finetuning.added_depth=${ADDED_DEPTH} \
        ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
        ++trainer.model_checkpoint.save_last=False \
    || echo "WARN: ${task} failed, continuing..."

    sleep 1
done
