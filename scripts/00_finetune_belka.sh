#!/usr/bin/env bash
# Fine-tune a pre-trained model on the 3 BELKA binding targets (subsampled).
# SKELETON — the BELKA data module is a draft; treat this as a template.
#
# Usage:
#   bash scripts/00_finetune_belka.sh <model> <checkpoint> [gpu_id]
#
# Notable env vars:
#   BELKA_SUBSAMPLE   : rows per target (default: 200000)
#   BELKA_SPLIT_KIND  : random | scaffold | kaggle_ood (default: random)
#   FINETUNE_DIM, UNFREEZE_DEPTH, EPOCH_UNFREEZE_ALL, ADDED_DEPTH:
#                       same meaning as in 00_finetune_admet.sh

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> <checkpoint> [gpu_id]"}
CKPT=${2:?   "Usage: $0 <model> <checkpoint> [gpu_id]"}
DEVICE=${3:-${DEVICE}}

FINETUNE_DIM=${FINETUNE_DIM:-256}
UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}
ADDED_DEPTH=${ADDED_DEPTH:-4}
BELKA_SUBSAMPLE=${BELKA_SUBSAMPLE:-200000}
BELKA_SPLIT_KIND=${BELKA_SPLIT_KIND:-random}

BELKA_TARGETS=(brd4 hsa seh)

PRETRAIN_DATASET=${PRETRAIN_DATASET:-auto}
if [[ "${PRETRAIN_DATASET}" == "auto" ]]; then
    CKPT_LOWER=$(echo "${CKPT}" | tr '[:upper:]' '[:lower:]')
    if [[ "${CKPT_LOWER}" == *"largemix"* ]]; then PRETRAIN_DATASET="largemix"
    else                                            PRETRAIN_DATASET="toymix"; fi
fi
FINETUNING_CONFIG=${FINETUNING_CONFIG:-belka}
TAGS="['${MODEL}','finetune','belka','${PRETRAIN_DATASET}']"

echo "=== Fine-tuning ${MODEL} on BELKA (ckpt=${CKPT}, split=${BELKA_SPLIT_KIND}, n=${BELKA_SUBSAMPLE}) ==="

for tgt in "${BELKA_TARGETS[@]}"; do
    task="belka_${tgt}"
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=belka \
        $(wandb_flags "${TAGS}") \
        ++constants.task=${task} \
        ++finetuning.task=${task} \
        ++datamodule.args.belka_targets=${tgt} \
        ++datamodule.args.subsample_size=${BELKA_SUBSAMPLE} \
        ++datamodule.args.split_kind=${BELKA_SPLIT_KIND} \
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
        ++trainer.model_checkpoint.dirpath=models_checkpoints/belka/${PRETRAIN_DATASET}/${MODEL}/${task}/ \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
