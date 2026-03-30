#!/usr/bin/env bash
# Pre-train a model, then fine-tune the resulting checkpoint on all 22 ADMET tasks.
#
# Usage:
#   bash scripts/03_pretrain_finetune.sh <model> <dataset> [gpu_id]
#
# Arguments:
#   model    : gcn | mpnn | gpspp | gpspp_800M | pairformer | pairformer_boltz
#   dataset  : any dataset accepted by 00_pretrain.sh
#   gpu_id   : CUDA device index (default: 0)
#
# Environment variables (optional — forwarded to the finetune stage):
#   FINETUNE_DIM         : hidden dim for finetuning head (default: 256)
#   UNFREEZE_DEPTH       : layers to unfreeze initially (default: 0)
#   EPOCH_UNFREEZE_ALL   : epoch to unfreeze all layers (default: none)
#   ADDED_DEPTH          : depth of the finetuning head (default: 4)
#   EXTRA_PRETRAIN_FLAGS : extra Hydra overrides for pre-training
#   EXTRA_FINETUNE_FLAGS : extra Hydra overrides for fine-tuning
#
# Examples:
#   bash scripts/03_pretrain_finetune.sh gpspp_800M largemix_dti_filtered 1
#   bash scripts/03_pretrain_finetune.sh pairformer_boltz toymix_dti_10k_filtered 5
#   UNFREEZE_DEPTH=4 bash scripts/03_pretrain_finetune.sh gpspp toymix 0

set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"

MODEL=${1:?  "Usage: $0 <model> <dataset> [gpu_id]"}
DATASET=${2:?  "Usage: $0 <model> <dataset> [gpu_id]"}
GPU_ID=${3:-${DEVICE:-0}}

echo "========================================================"
echo "  Stage 1: Pre-training ${MODEL} on ${DATASET} (GPU ${GPU_ID})"
echo "========================================================"

EXTRA_FLAGS="${EXTRA_PRETRAIN_FLAGS:-}" \
    bash "${SCRIPT_DIR}/00_pretrain.sh" "${MODEL}" "${DATASET}" "${GPU_ID}"

# ── Find the checkpoint produced by pre-training ──────────────────────────────
# The training/model config determines the checkpoint directory.
# Convention: models_checkpoints/<dataset-slug>/<model>/<timestamp>/last.ckpt
# We find the most recently modified last.ckpt under the expected tree.

# Map dataset to the directory slug used in checkpoint paths
case "${DATASET}" in
    toymix)                    CKPT_SLUG="small-dataset" ;;
    largemix)                  CKPT_SLUG="large-dataset" ;;
    toymix_dti)                CKPT_SLUG="toymix-dti" ;;
    largemix_dti)              CKPT_SLUG="largemix-dti" ;;
    toymix_dti_10k_filtered)   CKPT_SLUG="toymix-dti-10k-filtered" ;;
    largemix_dti_filtered)     CKPT_SLUG="largemix-dti-filtered" ;;
    toymix_rxrx3)              CKPT_SLUG="toymix-rxrx3" ;;
    toymix_rxrx3_dti)          CKPT_SLUG="toymix-rxrx3-dti" ;;
    largemix_rxrx3)            CKPT_SLUG="largemix-rxrx3" ;;
    largemix_rxrx3_dti)        CKPT_SLUG="largemix-rxrx3-dti" ;;
    rxrx3)                     CKPT_SLUG="rxrx3" ;;
    rxrx3_dti)                 CKPT_SLUG="rxrx3-dti" ;;
    dti)                       CKPT_SLUG="dti" ;;
    dti_filtered)              CKPT_SLUG="dti_filtered" ;;
    dti_10k_filtered)          CKPT_SLUG="dti_10k_filtered" ;;
    toymix_dti_filtered)       CKPT_SLUG="toymix-dti-filtered" ;;
    toymix_bbbc047)            CKPT_SLUG="toymix_bbbc047" ;;
    toymix_bbbc047_filtered)   CKPT_SLUG="toymix_bbbc047_filtered" ;;
    bbbc047)                   CKPT_SLUG="bbbc047" ;;
    *)                         CKPT_SLUG="${DATASET}" ;;
esac

CKPT_DIR="$(cd "${SCRIPT_DIR}/.."; pwd)/models_checkpoints/${CKPT_SLUG}/${MODEL}"
CKPT=$(find "${CKPT_DIR}" -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)

if [[ -z "${CKPT}" ]]; then
    echo "ERROR: No checkpoint found under ${CKPT_DIR}"
    echo "Pre-training may have failed. Check logs above."
    exit 1
fi

echo ""
echo "========================================================"
echo "  Stage 2: Fine-tuning on ADMET (checkpoint: ${CKPT})"
echo "========================================================"

EXTRA_FLAGS="${EXTRA_FINETUNE_FLAGS:-}" \
    bash "${SCRIPT_DIR}/01_finetune_admet.sh" "${MODEL}" "${CKPT}" "${GPU_ID}"
