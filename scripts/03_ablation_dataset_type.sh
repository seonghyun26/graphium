#!/usr/bin/env bash
# ============================================================================
# ABLATION STUDY: Pre-training Dataset Type
# ============================================================================
# Tests how the TYPE of pre-training data affects downstream ADMET performance.
#
# Fixed variables:
#   - Model architecture: GPS++ (configurable via MODEL)
#   - Model size: same DIM/GNN_DEPTH across all runs
#   - Finetuning protocol: identical for all conditions
#   - Random seed: fixed
#
# Ablated variable:
#   - Pre-training dataset: {toymix, largemix, rxrx3, largemix_rxrx3}
#   - Plus a "scratch" baseline (no pre-training)
#
# Pipeline per condition:
#   1. Pre-train on dataset D  (00_pretrain.sh)
#   2. Fine-tune on ADMET      (01_finetune_admet.sh)
#
# Usage:
#   bash scripts/03_ablation_dataset_type.sh [gpu_id]
#
# Environment variables (optional):
#   MODEL      : model architecture (default: gpspp)
#   DIM        : hidden dimension (default: 1472)
#   GNN_DEPTH  : GNN layers (default: 8)
#   BATCH_SIZE : pre-training batch size (default: 300)
#   CKPT_DIR   : directory to store checkpoints (default: ./ablation_checkpoints)
#
# Output:
#   Checkpoints saved to: ${CKPT_DIR}/type_ablation/${MODEL}_${DATASET}/
#   W&B tags include: ablation, dataset_type

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}

# ── Ablation configuration ───────────────────────────────────────────────────
MODEL=${MODEL:-gpspp}
DIM=${DIM:-1472}
GNN_DEPTH=${GNN_DEPTH:-8}
BATCH_SIZE=${BATCH_SIZE:-300}
CKPT_DIR=${CKPT_DIR:-./ablation_checkpoints}

DATASETS=(toymix largemix rxrx3 largemix_rxrx3)

FINETUNE_DIM=256
UNFREEZE_DEPTH=0
EPOCH_UNFREEZE_ALL=none
FINETUNING_CONFIG=admet

echo "============================================================"
echo "  ABLATION: Pre-training Dataset Type"
echo "  Model: ${MODEL}  Dim: ${DIM}  Depth: ${GNN_DEPTH}"
echo "  Datasets: ${DATASETS[*]}"
echo "============================================================"

# ── Step 0: Scratch baseline ─────────────────────────────────────────────────
echo ""
echo ">>> [0/5] Scratch baseline (no pre-training) <<<"
EXTRA_FLAGS="++constants.wandb.tags=\"['${MODEL}','scratch','admet','ablation','dataset_type']\"" \
    bash "$(dirname "$0")/02_scratch_admet.sh" "${MODEL}" "${DEVICE}"

# ── Step 1-4: Pre-train + Fine-tune for each dataset ────────────────────────
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    STEP=$((i + 1))
    TOTAL=${#DATASETS[@]}

    echo ""
    echo ">>> [${STEP}/${TOTAL}] Pre-training on ${DATASET} <<<"

    # Pre-train
    CKPT_PATH="${CKPT_DIR}/type_ablation/${MODEL}_${DATASET}"
    mkdir -p "${CKPT_PATH}"

    DEVICE=${DEVICE} DIM=${DIM} GNN_DEPTH=${GNN_DEPTH} BATCH_SIZE=${BATCH_SIZE} \
    EXTRA_FLAGS="++constants.wandb.tags=\"['${MODEL}','pretrain','${DATASET}','ablation','dataset_type']\" \
        ++trainer.model_checkpoint.dirpath=${CKPT_PATH} \
        ++trainer.model_checkpoint.save_last=True" \
        bash "$(dirname "$0")/00_pretrain.sh" "${MODEL}" "${DATASET}" "${DEVICE}"

    echo ""
    echo ">>> [${STEP}/${TOTAL}] Fine-tuning ${DATASET}-pretrained on ADMET <<<"

    # Fine-tune
    CKPT_FILE="${CKPT_PATH}/last.ckpt"
    if [[ ! -f "${CKPT_FILE}" ]]; then
        echo "WARN: checkpoint not found at ${CKPT_FILE}, skipping fine-tuning for ${DATASET}"
        continue
    fi

    DEVICE=${DEVICE} \
    FINETUNE_DIM=${FINETUNE_DIM} \
    UNFREEZE_DEPTH=${UNFREEZE_DEPTH} \
    EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL} \
    FINETUNING_CONFIG=${FINETUNING_CONFIG} \
    EXTRA_FLAGS="++constants.wandb.tags=\"['${MODEL}','finetune','admet','${DATASET}','ablation','dataset_type']\"" \
        bash "$(dirname "$0")/01_finetune_admet.sh" "${MODEL}" "${CKPT_FILE}" "${DEVICE}"
done

echo ""
echo "============================================================"
echo "  ABLATION COMPLETE: Dataset Type"
echo "  Results logged to W&B project: ${WANDB_PROJECT}"
echo "  Filter by tag: ablation, dataset_type"
echo "============================================================"
