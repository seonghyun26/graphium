#!/usr/bin/env bash
# Pre-train GPS++ 800M with MoE on toymix+dti.
#
# Usage:
#   bash scripts/06_pretrain_moe_800M.sh [gpu_id]
#
# Uses the gpspp_800M_moe model config which inherits gpspp_800M and adds
# MoE (4 experts, top-2) to the FFN layers.

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}
DATASET=toymix_dti_filtered

TAGS="['gpspp_800M','moe','pretrain','${DATASET}']"

echo "=== Pre-training GPS++ 800M + MoE on ${DATASET} ==="
echo "  GPU: ${DEVICE}"
echo ""

CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
    model=gpspp_800M_moe \
    accelerator=gpu \
    tasks=${DATASET} \
    training=${DATASET} \
    architecture=toymix \
    $(wandb_flags "${TAGS}") \
    ++constants.norm=layer_norm \
    ++architecture.gnn.depth=12 \
    ++trainer.trainer.precision=bf16-mixed \
    ${EXTRA_FLAGS:-}

echo ""
echo "=== Done: GPS++ 800M + MoE pre-training ==="
