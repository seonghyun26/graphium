#!/usr/bin/env bash
# ============================================================================
# ABLATION STUDY: Pre-training Dataset Size
# ============================================================================
# Tests how the SIZE of the pre-training data affects downstream ADMET
# performance, using the sample_size parameter to subsample the dataset.
#
# Fixed variables:
#   - Model architecture: GPS++ (configurable via MODEL)
#   - Model size: same DIM/GNN_DEPTH across all runs
#   - Pre-training dataset type: largemix (configurable via DATASET)
#   - Finetuning protocol: identical for all conditions
#   - Random seed: fixed
#
# Ablated variable:
#   - Data fraction: {0.01, 0.05, 0.1, 0.25, 0.5, 1.0}
#     (applied uniformly to all sub-tasks in the pre-training dataset)
#
# Pipeline per condition:
#   1. Pre-train on fraction f of dataset  (00_pretrain.sh + SAMPLE_SIZE)
#   2. Fine-tune on ADMET                  (01_finetune_admet.sh)
#
# Usage:
#   bash scripts/04_ablation_dataset_size.sh [gpu_id]
#
# Environment variables (optional):
#   MODEL      : model architecture (default: gpspp)
#   DATASET    : pre-training dataset to subsample (default: largemix)
#   DIM        : hidden dimension (default: 1472)
#   GNN_DEPTH  : GNN layers (default: 8)
#   BATCH_SIZE : pre-training batch size (default: 300)
#   FRACTIONS  : space-separated fractions (default: "0.01 0.05 0.1 0.25 0.5 1.0")
#   CKPT_DIR   : directory to store checkpoints (default: ./ablation_checkpoints)
#
# Output:
#   Checkpoints saved to: ${CKPT_DIR}/size_ablation/${MODEL}_${DATASET}_frac${FRAC}/
#   W&B tags include: ablation, dataset_size

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}

# ── Ablation configuration ───────────────────────────────────────────────────
MODEL=${MODEL:-gpspp}
DATASET=${DATASET:-largemix}
DIM=${DIM:-1472}
GNN_DEPTH=${GNN_DEPTH:-8}
BATCH_SIZE=${BATCH_SIZE:-300}
CKPT_DIR=${CKPT_DIR:-./ablation_checkpoints}

FRACTIONS_STR=${FRACTIONS:-"0.01 0.05 0.1 0.25 0.5 1.0"}
read -ra FRACTIONS <<< "${FRACTIONS_STR}"

FINETUNE_DIM=256
UNFREEZE_DEPTH=4
EPOCH_UNFREEZE_ALL=40
FINETUNING_CONFIG=admet

echo "============================================================"
echo "  ABLATION: Pre-training Dataset Size"
echo "  Model: ${MODEL}  Dataset: ${DATASET}"
echo "  Dim: ${DIM}  Depth: ${GNN_DEPTH}"
echo "  Fractions: ${FRACTIONS[*]}"
echo "============================================================"

# ── Step 0: Scratch baseline ─────────────────────────────────────────────────
echo ""
echo ">>> [0/${#FRACTIONS[@]}] Scratch baseline (no pre-training) <<<"
EXTRA_FLAGS="++constants.wandb.tags=\"['${MODEL}','scratch','admet','ablation','dataset_size']\"" \
    bash "$(dirname "$0")/02_scratch_admet.sh" "${MODEL}" "${DEVICE}"

# ── Pre-train + Fine-tune for each fraction ──────────────────────────────────
for i in "${!FRACTIONS[@]}"; do
    FRAC="${FRACTIONS[$i]}"
    STEP=$((i + 1))
    TOTAL=${#FRACTIONS[@]}

    echo ""
    echo ">>> [${STEP}/${TOTAL}] Pre-training on ${DATASET} with fraction=${FRAC} <<<"

    # Pre-train with subsampled data
    CKPT_PATH="${CKPT_DIR}/size_ablation/${MODEL}_${DATASET}_frac${FRAC}"
    mkdir -p "${CKPT_PATH}"

    DEVICE=${DEVICE} DIM=${DIM} GNN_DEPTH=${GNN_DEPTH} BATCH_SIZE=${BATCH_SIZE} \
    SAMPLE_SIZE=${FRAC} \
    EXTRA_FLAGS="++constants.wandb.tags=\"['${MODEL}','pretrain','${DATASET}','frac_${FRAC}','ablation','dataset_size']\" \
        ++trainer.model_checkpoint.dirpath=${CKPT_PATH} \
        ++trainer.model_checkpoint.save_last=True" \
        bash "$(dirname "$0")/00_pretrain.sh" "${MODEL}" "${DATASET}" "${DEVICE}"

    echo ""
    echo ">>> [${STEP}/${TOTAL}] Fine-tuning frac=${FRAC}-pretrained on ADMET <<<"

    # Fine-tune
    CKPT_FILE="${CKPT_PATH}/last.ckpt"
    if [[ ! -f "${CKPT_FILE}" ]]; then
        echo "WARN: checkpoint not found at ${CKPT_FILE}, skipping fine-tuning for frac=${FRAC}"
        continue
    fi

    DEVICE=${DEVICE} \
    FINETUNE_DIM=${FINETUNE_DIM} \
    UNFREEZE_DEPTH=${UNFREEZE_DEPTH} \
    EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL} \
    FINETUNING_CONFIG=${FINETUNING_CONFIG} \
    EXTRA_FLAGS="++constants.wandb.tags=\"['${MODEL}','finetune','admet','${DATASET}','frac_${FRAC}','ablation','dataset_size']\"" \
        bash "$(dirname "$0")/01_finetune_admet.sh" "${MODEL}" "${CKPT_FILE}" "${DEVICE}"
done

echo ""
echo "============================================================"
echo "  ABLATION COMPLETE: Dataset Size"
echo "  Results logged to W&B project: ${WANDB_PROJECT}"
echo "  Filter by tags: ablation, dataset_size"
echo "============================================================"
