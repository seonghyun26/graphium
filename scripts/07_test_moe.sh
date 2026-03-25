#!/usr/bin/env bash
# Test Mixture-of-Experts on Pairformer with a small model on ToyMix.
#
# Usage:
#   bash scripts/07_test_moe.sh [gpu_id]
#
# This runs a quick debug training (3 epochs, 10 batches) with a small
# Pairformer (dim=128, depth=2) using MoE on the single-track transition
# (4 experts, top-2 routing). Compares against a baseline without MoE.
#
# The model is small enough to fit on any GPU.

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}

echo "=== MoE Test: Pairformer on ToyMix (GPU ${DEVICE}) ==="
echo ""

# ── Common settings ──────────────────────────────────────────────────────────
DIM=128
DEPTH=2
BATCH_SIZE=16
EPOCHS=3
LIMIT_BATCHES=10

COMMON_FLAGS="\
    model=pairformer \
    accelerator=gpu \
    tasks=toymix \
    training=toymix \
    architecture=toymix \
    ++constants.seed=${SEED} \
    ++constants.norm=layer_norm \
    ++constants.max_epochs=${EPOCHS} \
    ++constants.raise_train_error=true \
    ++constants.name=moe_test \
    ++trainer.trainer.limit_train_batches=${LIMIT_BATCHES} \
    ++trainer.trainer.limit_val_batches=3 \
    ++trainer.trainer.check_val_every_n_epoch=1 \
    ++trainer.trainer.precision=bf16-mixed \
    ++architecture.pre_nn.out_dim=${DIM} \
    ++architecture.pre_nn.hidden_dims=${DIM} \
    ++architecture.gnn.in_dim=${DIM} \
    ++architecture.gnn.out_dim=${DIM} \
    ++architecture.gnn.hidden_dims=${DIM} \
    ++architecture.gnn.depth=${DEPTH} \
    ++architecture.graph_output_nn.graph.hidden_dims=${DIM} \
    ++datamodule.args.batch_size_training=${BATCH_SIZE} \
    ++datamodule.args.num_workers=0 \
    ++datamodule.args.persistent_workers=false \
    ++datamodule.args.featurization_n_jobs=0 \
    ++trainer.model_checkpoint.save_last=False \
    ++constants.wandb.entity=eddy26 \
    ++constants.wandb.project=graphium"

# ── Run 1: Baseline (no MoE) ────────────────────────────────────────────────
echo "--- Run 1: Baseline (no MoE) ---"

CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
    ${COMMON_FLAGS} \
    "++constants.wandb.tags=['moe_test','baseline']" \
    2>&1 | tee /tmp/moe_test_baseline.log | tail -20

echo ""
echo "--- Run 2: MoE (4 experts, top-2) ---"

# ── Run 2: MoE (4 experts, top-2) ───────────────────────────────────────────
CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
    ${COMMON_FLAGS} \
    ++architecture.gnn.layer_kwargs.moe_num_experts=4 \
    ++architecture.gnn.layer_kwargs.moe_top_k=2 \
    ++architecture.gnn.layer_kwargs.moe_aux_loss_coeff=0.01 \
    "++constants.wandb.tags=['moe_test','moe_4exp_top2']" \
    2>&1 | tee /tmp/moe_test_moe.log | tail -20

echo ""
echo "=== Comparison ==="
echo "Baseline loss:"
grep "loss/test" /tmp/moe_test_baseline.log | tail -1
echo "MoE loss:"
grep "loss/test" /tmp/moe_test_moe.log | tail -1
