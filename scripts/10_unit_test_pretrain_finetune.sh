#!/usr/bin/env bash
# Unit test: short pre-train + finetune on 5 representative ADMET tasks (one per A/D/M/E/T).
#
# Usage:
#   bash scripts/10_unit_test_pretrain_finetune.sh <model> [gpu_id]
#
# Arguments:
#   model    : gcn | mpnn | gpspp | gpspp_800M | pairformer_boltz | ...
#   gpu_id   : CUDA device index (default: 0)
#
# Environment variables (optional):
#   PRETRAIN_EPOCHS      : pre-training epochs (default: 3)
#   PRETRAIN_BATCHES     : train batches per epoch (default: 10)
#   FINETUNE_EPOCHS      : fine-tuning epochs (default: 3)
#   FINETUNE_BATCHES     : train batches per epoch (default: 10)
#   PRETRAIN_DATASET     : dataset for pre-training (default: toymix)
#
# Examples:
#   bash scripts/10_unit_test_pretrain_finetune.sh gpspp_800M 5
#   PRETRAIN_EPOCHS=5 bash scripts/10_unit_test_pretrain_finetune.sh pairformer_boltz 2

set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"

MODEL=${1:?  "Usage: $0 <model> [gpu_id]"}
GPU_ID=${2:-${DEVICE:-0}}

PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-3}
PRETRAIN_BATCHES=${PRETRAIN_BATCHES:-10}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-3}
FINETUNE_BATCHES=${FINETUNE_BATCHES:-10}
PRETRAIN_DATASET=${PRETRAIN_DATASET:-toymix}

# 5 representative ADMET tasks — one per category
export ADMET_TASKS_OVERRIDE="caco2_wang bbb_martins cyp2d6_veith clearance_hepatocyte_az ld50_zhu"

echo "========================================================"
echo "  Unit Test: ${MODEL} on ${PRETRAIN_DATASET} → 5 ADMET tasks"
echo "  Pre-train: ${PRETRAIN_EPOCHS} epochs, ${PRETRAIN_BATCHES} batches"
echo "  Fine-tune: ${FINETUNE_EPOCHS} epochs, ${FINETUNE_BATCHES} batches"
echo "  Tasks: ${ADMET_TASKS_OVERRIDE}"
echo "  GPU: ${GPU_ID}"
echo "========================================================"

# ── Stage 1: Short pre-training ─────────────────────────────────────────────
echo ""
echo "=== Stage 1: Pre-training ${MODEL} on ${PRETRAIN_DATASET} ==="

EXTRA_FLAGS="++constants.max_epochs=${PRETRAIN_EPOCHS} \
    ++constants.raise_train_error=true \
    ++trainer.trainer.limit_train_batches=${PRETRAIN_BATCHES} \
    ++trainer.trainer.limit_val_batches=3 \
    ++trainer.trainer.check_val_every_n_epoch=1" \
    bash "${SCRIPT_DIR}/00_pretrain.sh" "${MODEL}" "${PRETRAIN_DATASET}" "${GPU_ID}"

# ── Find the checkpoint ─────────────────────────────────────────────────────
case "${PRETRAIN_DATASET}" in
    toymix)                    CKPT_SLUG="small-dataset" ;;
    largemix)                  CKPT_SLUG="large-dataset" ;;
    toymix_dti)                CKPT_SLUG="toymix-dti" ;;
    toymix_dti_filtered)       CKPT_SLUG="toymix-dti-filtered" ;;
    toymix_bbbc047)            CKPT_SLUG="toymix_bbbc047" ;;
    toymix_bbbc047_filtered)   CKPT_SLUG="toymix_bbbc047_filtered" ;;
    *)                         CKPT_SLUG="${PRETRAIN_DATASET}" ;;
esac

CKPT_DIR="$(cd "${SCRIPT_DIR}/.."; pwd)/models_checkpoints/${CKPT_SLUG}/${MODEL}"
CKPT=$(find "${CKPT_DIR}" -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)

if [[ -z "${CKPT}" ]]; then
    echo "ERROR: No checkpoint found under ${CKPT_DIR}"
    exit 1
fi

echo ""
echo "=== Stage 2: Fine-tuning on 5 ADMET tasks (ckpt: ${CKPT}) ==="

# ── Stage 2: Fine-tune on 5 tasks ───────────────────────────────────────────
EXTRA_FLAGS="++constants.max_epochs=${FINETUNE_EPOCHS} \
    ++constants.raise_train_error=false \
    ++trainer.trainer.limit_train_batches=${FINETUNE_BATCHES} \
    ++trainer.trainer.limit_val_batches=3 \
    ++trainer.trainer.check_val_every_n_epoch=1" \
    bash "${SCRIPT_DIR}/01_finetune_admet.sh" "${MODEL}" "${CKPT}" "${GPU_ID}"

echo ""
echo "========================================================"
echo "  Unit test complete: ${MODEL}"
echo "========================================================"
