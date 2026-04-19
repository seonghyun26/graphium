#!/usr/bin/env bash
# Pre-train, then fine-tune on the 22 ADMET tasks using the resulting
# checkpoint. Runs both stages on the same GPU sequentially; fine-tune
# only runs if pre-train exits 0.
#
# Usage:
#   bash scripts/00_pretrain_then_finetune_admet.sh <model> <dataset> [gpu_id]
#
# Same <model>/<dataset> grammar as 00_pretrain.sh.
#
# Env overrides forwarded to each stage:
#   BATCH_SIZE, SAMPLE_SIZE, EXTRA_FLAGS            → pretrain
#   FINETUNE_DIM, UNFREEZE_DEPTH, EPOCH_UNFREEZE_ALL,
#   FINETUNING_CONFIG, SUB_MODULE, USE_COSINE,
#   ADDED_DEPTH, MODEL_TAG                          → finetune
#
# Example:
#   bash scripts/00_pretrain_then_finetune_admet.sh pairmixer_12M toymix 3

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> <dataset> [gpu_id]"}
DATASET=${2:?  "Usage: $0 <model> <dataset> [gpu_id]"}
DEVICE=${3:-${DEVICE}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR="$(pwd)/models_checkpoints/chain/${MODEL}_${DATASET}_${STAMP}"

echo "=== [1/2] Pre-train → checkpoint dir: ${CKPT_DIR} ==="
EXTRA_FLAGS="${EXTRA_FLAGS:-} ++trainer.model_checkpoint.dirpath=${CKPT_DIR}/" \
    bash "${SCRIPT_DIR}/00_pretrain.sh" "${MODEL}" "${DATASET}" "${DEVICE}"

CKPT="${CKPT_DIR}/last.ckpt"
if [[ ! -f "${CKPT}" ]]; then
    echo "Error: expected checkpoint not found at ${CKPT}" >&2
    exit 1
fi

echo "=== [2/2] Fine-tune on ADMET → ckpt: ${CKPT} ==="
# Pin PRETRAIN_DATASET so finetune does not mis-infer it from our pinned path.
PRETRAIN_DATASET="${DATASET}" \
    bash "${SCRIPT_DIR}/00_finetune_admet.sh" "${MODEL}" "${CKPT}" "${DEVICE}"
