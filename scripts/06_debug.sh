#!/usr/bin/env bash
# Quick sanity-check: run a small model for a few epochs.
# Usage: bash scripts/06_debug.sh [model] [dataset] [gpu_id]

source "$(dirname "$0")/common.sh"

MODEL=${1:-gpspp}
DATASET=${2:-toymix}
DEVICE=${3:-${DEVICE}}

echo "=== Debug run: ${MODEL} on ${DATASET} (3 epochs, 10 batches) ==="

DIM=128 GNN_DEPTH=2 BATCH_SIZE=16 SEED=42 \
EXTRA_FLAGS="++constants.name=${MODEL}_debug \
    ++constants.max_epochs=3 \
    ++constants.raise_train_error=true \
    ++trainer.trainer.limit_train_batches=10 \
    ++trainer.trainer.limit_val_batches=3 \
    ++trainer.trainer.check_val_every_n_epoch=1" \
    bash "$(dirname "$0")/00_pretrain.sh" "${MODEL}" "${DATASET}" "${DEVICE}"
