#!/usr/bin/env bash
# Fine-tune a pre-trained model on the 29-assay cell_bioactivity benchmark
# (Fredinh et al., Nat. Commun. 2024).
#
# Unlike 00_finetune_admet.sh which loops over 22 TDC tasks, this is a single
# multi-output task: one graphium-train invocation with task=cell_bioactivity.
#
# Usage:
#   bash scripts/00_finetune_cell_bioactivity.sh <model> <checkpoint> [gpu_id]
#
# Arguments:
#   model      : gcn | mpnn | gpspp | gpspp_800M | pairmixer_* | ...
#   checkpoint : path to pre-trained .ckpt file
#   gpu_id     : CUDA device index (default: DEVICE env, then 0)
#
# Environment variables (optional):
#   FINETUNE_DIM         : hidden dim for finetuning head (default: 256)
#   UNFREEZE_DEPTH       : layers to unfreeze initially (default: 0, frozen backbone)
#   EPOCH_UNFREEZE_ALL   : epoch to unfreeze all layers (default: none)
#   ADDED_DEPTH          : finetuning head depth (default: 4)
#   SUB_MODULE           : override sub_module_from_pretrained (default: zinc)
#   EXTRA_FLAGS          : any additional Hydra CLI flags
#
# Example:
#   bash scripts/00_finetune_cell_bioactivity.sh gpspp ./ckpt.ckpt 5

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> <checkpoint> [gpu_id]"}
CKPT=${2:?   "Usage: $0 <model> <checkpoint> [gpu_id]"}
DEVICE=${3:-${DEVICE}}

FINETUNE_DIM=${FINETUNE_DIM:-256}
UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}
ADDED_DEPTH=${ADDED_DEPTH:-4}

if [[ ! -f "${CKPT}" ]]; then
    echo "Error: checkpoint not found: ${CKPT}"
    exit 1
fi

TAGS="['${MODEL}','finetune','cell_bioactivity']"

echo "=== Fine-tuning ${MODEL} on cell_bioactivity (ckpt=${CKPT}, unfreeze=${UNFREEZE_DEPTH}) ==="

CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
    model=${MODEL} \
    accelerator=gpu \
    tasks=cell_bioactivity \
    $(wandb_flags "${TAGS}") \
    +finetuning=cell_bioactivity \
    ++finetuning.pretrained_model=${CKPT} \
    ++finetuning.unfreeze_pretrained_depth=${UNFREEZE_DEPTH} \
    ++finetuning.epoch_unfreeze_all=${EPOCH_UNFREEZE_ALL} \
    ++finetuning.finetuning_head.in_dim=${FINETUNE_DIM} \
    ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
    ++finetuning.new_out_dim=${FINETUNE_DIM} \
    ++finetuning.finetuning_head.depth=${ADDED_DEPTH} \
    ++finetuning.added_depth=${ADDED_DEPTH} \
    ++architecture.task_heads.cell_bioactivity.hidden_dims=${FINETUNE_DIM} \
    ++trainer.model_checkpoint.dirpath=models_checkpoints/cell_bioactivity/${MODEL}/ \
    ${SUB_MODULE:+++finetuning.sub_module_from_pretrained=${SUB_MODULE}} \
    ${EXTRA_FLAGS:-}
