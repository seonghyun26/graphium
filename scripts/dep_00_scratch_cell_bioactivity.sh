#!/usr/bin/env bash
# Train a model from scratch on the 29-assay cell_bioactivity benchmark
# (Fredinh et al., Nat. Commun. 2024). No pre-trained checkpoint — weights are
# initialized fresh and the full backbone + head train together.
#
# Mirrors 00_finetune_cell_bioactivity.sh but strips the `+finetuning=...` path
# and bakes the cell_bioactivity training protocol (lr=1e-4, cosine to 1e-6,
# 50 epochs) inline so the (training,model) defaults (toymix_pairmixer_12M etc.)
# only contribute batch-size / trainer / checkpoint layout.
#
# Usage:
#   bash scripts/00_scratch_cell_bioactivity.sh <model> [gpu_id]
#
# Arguments:
#   model   : gcn | mpnn | gpspp | gpspp_800M | pairmixer_* | ...
#   gpu_id  : CUDA device index (default: DEVICE env, then 0)
#
# Environment variables (optional):
#   MAX_EPOCHS   : override training length (default: 50)
#   LR           : override optimizer lr (default: 1e-4)
#   BATCH_SIZE   : override datamodule batch_size (default: inherited from training/model yaml)
#   EXTRA_FLAGS  : any additional Hydra CLI flags
#
# Example:
#   bash scripts/00_scratch_cell_bioactivity.sh pairmixer_12M 7

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> [gpu_id]"}
DEVICE=${2:-${DEVICE}}

MAX_EPOCHS=${MAX_EPOCHS:-50}

TAGS="['${MODEL}','scratch','cell_bioactivity']"

BATCH_FLAGS=""
if [[ -n "${BATCH_SIZE:-}" ]]; then
    BATCH_FLAGS="++datamodule.args.batch_size_training=${BATCH_SIZE} ++datamodule.args.batch_size_inference=${BATCH_SIZE}"
fi

# Keep the proven-safe toymix training recipe (lr=4e-5 + WarmUpLinearLR) —
# overriding it with 1e-4/CosineAnnealingLR caused the pairmixer_12M backbone
# to emit NaN logits within a couple of epochs (probe AUROC stuck at 0.499 all
# run). Only trim max_epochs to match the cell_bioactivity 50-epoch protocol.
echo "=== Training ${MODEL} from scratch on cell_bioactivity (epochs=${MAX_EPOCHS} gpu=${DEVICE}) ==="

CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
    model=${MODEL} \
    accelerator=gpu \
    tasks=cell_bioactivity \
    $(wandb_flags "${TAGS}") \
    ++constants.name=scratch_cell_bioactivity_${MODEL} \
    ++constants.max_epochs=${MAX_EPOCHS} \
    ++constants.raise_train_error=false \
    ++trainer.model_checkpoint.dirpath=models_checkpoints/cell_bioactivity_scratch/${MODEL}/ \
    ++trainer.model_checkpoint.save_last=false \
    ${BATCH_FLAGS} \
    ${EXTRA_FLAGS:-}
