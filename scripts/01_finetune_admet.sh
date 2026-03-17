#!/usr/bin/env bash
# Fine-tune a pre-trained model on all 22 ADMET benchmark tasks.
#
# Usage:
#   bash scripts/01_finetune_admet.sh <model> <checkpoint> [gpu_id]
#
# Arguments:
#   model      : gcn | mpnn | gpspp
#   checkpoint : path to pre-trained .ckpt file
#   gpu_id     : CUDA device index (default: 0)
#
# Environment variables (optional):
#   FINETUNE_DIM         : hidden dim for finetuning head (default: 256)
#   UNFREEZE_DEPTH       : layers to unfreeze initially (default: 0, frozen backbone)
#   EPOCH_UNFREEZE_ALL   : epoch to unfreeze all layers (default: none, stay frozen)
#   FINETUNING_CONFIG    : hydra finetuning config (default: admet)
#   SUB_MODULE           : sub_module_from_pretrained (default: auto-detect)
#
# Examples:
#   bash scripts/01_finetune_admet.sh gpspp ./checkpoints/gpspp_largemix.ckpt 0
#   UNFREEZE_DEPTH=4 EPOCH_UNFREEZE_ALL=40 bash scripts/01_finetune_admet.sh gcn ./ckpt.ckpt 1

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> <checkpoint> [gpu_id]"}
CKPT=${2:?   "Usage: $0 <model> <checkpoint> [gpu_id]"}
DEVICE=${3:-${DEVICE}}

FINETUNE_DIM=${FINETUNE_DIM:-256}
UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}
FINETUNING_CONFIG=${FINETUNING_CONFIG:-admet}
ADDED_DEPTH=${ADDED_DEPTH:-4}

# ── Determine sub_module and dim flags based on model ────────────────────────
case "${MODEL}" in
    gcn)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    mpnn)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    gpspp)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    *)
        echo "Error: unknown model '${MODEL}'."
        exit 1
        ;;
esac

TAGS="['${MODEL}','finetune','admet']"

echo "=== Fine-tuning ${MODEL} on ADMET (ckpt=${CKPT}, unfreeze=${UNFREEZE_DEPTH}) ==="

for task in "${ADMET_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
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
        ++finetuning.unfreeze_pretrained_depth=${UNFREEZE_DEPTH} \
        ++finetuning.epoch_unfreeze_all=${EPOCH_UNFREEZE_ALL} \
        ++finetuning.finetuning_head.in_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
        ++finetuning.new_out_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.depth=${ADDED_DEPTH} \
        ++finetuning.added_depth=${ADDED_DEPTH} \
        ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
        ++trainer.model_checkpoint.save_last=False \
        ${SUB_MODULE:+++finetuning.sub_module_from_pretrained=${SUB_MODULE}} \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
