#!/usr/bin/env bash
# ============================================================================
# ABLATION STUDY: Full Matrix (Dataset Type x Size)
# ============================================================================
# Runs the complete ablation over both dataset TYPE and SIZE jointly.
# This produces the full results table for the paper.
#
# Matrix:
#   Rows    = dataset type: {toymix, largemix, rxrx3, largemix_rxrx3}
#   Columns = dataset fraction: {0.1, 0.25, 0.5, 1.0}
#   Cell    = pretrain(type, frac) -> finetune(ADMET 22 tasks)
#   + 1 scratch baseline row (no pre-training)
#
# Total runs: 1 (scratch) + 4 types x 4 fractions = 17 pretraining runs
#             + 17 x 22 ADMET finetuning runs = 374 finetuning runs
#
# Usage:
#   bash scripts/05_ablation_full_matrix.sh [gpu_id]
#
# Environment variables:
#   MODEL, DIM, GNN_DEPTH, BATCH_SIZE  - see 00_pretrain.sh
#   CKPT_DIR  - checkpoint root (default: ./ablation_checkpoints)

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}

MODEL=${MODEL:-gpspp}
DIM=${DIM:-1472}
GNN_DEPTH=${GNN_DEPTH:-8}
BATCH_SIZE=${BATCH_SIZE:-300}
CKPT_DIR=${CKPT_DIR:-./ablation_checkpoints}

DATASETS=(toymix largemix rxrx3 largemix_rxrx3)
FRACTIONS=(0.1 0.25 0.5 1.0)

FINETUNE_DIM=256
UNFREEZE_DEPTH=0
EPOCH_UNFREEZE_ALL=none
FINETUNING_CONFIG=admet

TOTAL_PRETRAIN=$(( ${#DATASETS[@]} * ${#FRACTIONS[@]} ))
RUN=0

echo "============================================================"
echo "  FULL ABLATION MATRIX: Dataset Type x Size"
echo "  Model: ${MODEL}  Dim: ${DIM}  Depth: ${GNN_DEPTH}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Fractions: ${FRACTIONS[*]}"
echo "  Total pre-training runs: ${TOTAL_PRETRAIN} + 1 scratch"
echo "============================================================"

# ── Scratch baseline ─────────────────────────────────────────────────────────
echo ""
echo ">>> [scratch] Baseline: no pre-training <<<"
EXTRA_FLAGS="++constants.wandb.tags=\"['${MODEL}','scratch','admet','ablation','full_matrix']\"" \
    bash "$(dirname "$0")/00_scratch_admet.sh" "${MODEL}" "${DEVICE}"

# ── Full matrix ──────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    for FRAC in "${FRACTIONS[@]}"; do
        RUN=$((RUN + 1))

        echo ""
        echo ">>> [${RUN}/${TOTAL_PRETRAIN}] ${DATASET} x frac=${FRAC} <<<"

        CKPT_PATH="${CKPT_DIR}/full_matrix/${MODEL}_${DATASET}_frac${FRAC}"
        mkdir -p "${CKPT_PATH}"

        # Pre-train
        DEVICE=${DEVICE} DIM=${DIM} GNN_DEPTH=${GNN_DEPTH} BATCH_SIZE=${BATCH_SIZE} \
        SAMPLE_SIZE=${FRAC} \
        EXTRA_FLAGS="++constants.wandb.tags=\"['${MODEL}','pretrain','${DATASET}','frac_${FRAC}','ablation','full_matrix']\" \
            ++trainer.model_checkpoint.dirpath=${CKPT_PATH} \
            ++trainer.model_checkpoint.save_last=True" \
            bash "$(dirname "$0")/00_pretrain.sh" "${MODEL}" "${DATASET}" "${DEVICE}"

        # Fine-tune
        CKPT_FILE="${CKPT_PATH}/last.ckpt"
        if [[ ! -f "${CKPT_FILE}" ]]; then
            echo "WARN: checkpoint not found at ${CKPT_FILE}, skipping"
            continue
        fi

        DEVICE=${DEVICE} \
        FINETUNE_DIM=${FINETUNE_DIM} \
        UNFREEZE_DEPTH=${UNFREEZE_DEPTH} \
        EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL} \
        FINETUNING_CONFIG=${FINETUNING_CONFIG} \
        EXTRA_FLAGS="++constants.wandb.tags=\"['${MODEL}','finetune','admet','${DATASET}','frac_${FRAC}','ablation','full_matrix']\"" \
            bash "$(dirname "$0")/00_finetune_admet.sh" "${MODEL}" "${CKPT_FILE}" "${DEVICE}"
    done
done

echo ""
echo "============================================================"
echo "  FULL ABLATION MATRIX COMPLETE"
echo "  Results logged to W&B project: ${WANDB_PROJECT}"
echo "  Filter by tags: ablation, full_matrix"
echo "============================================================"
