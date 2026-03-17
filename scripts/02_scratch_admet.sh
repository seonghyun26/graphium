#!/usr/bin/env bash
# Train a model from scratch on ADMET benchmark tasks (no pre-training).
# This serves as the baseline for the dataset ablation study.
#
# Usage:
#   bash scripts/02_scratch_admet.sh <model> [gpu_id]
#
# Arguments:
#   model  : gcn | mpnn | gpspp
#   gpu_id : CUDA device index (default: 0)
#
# Environment variables (optional):
#   DIM        : hidden dimension (default: model-specific)
#   GNN_DEPTH  : GNN depth (default: 16)
#   BATCH_SIZE : batch size (default: 400)
#
# Examples:
#   bash scripts/02_scratch_admet.sh gpspp 0
#   DIM=1024 bash scripts/02_scratch_admet.sh gcn 1

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> [gpu_id]"}
DEVICE=${2:-${DEVICE}}

# ── Model-specific defaults ──────────────────────────────────────────────────
case "${MODEL}" in
    gcn)
        DIM=${DIM:-5120}
        GNN_DEPTH=${GNN_DEPTH:-16}
        BATCH_SIZE=${BATCH_SIZE:-400}
        ;;
    mpnn)
        DIM=${DIM:-400}
        GNN_DEPTH=${GNN_DEPTH:-16}
        BATCH_SIZE=${BATCH_SIZE:-400}
        ;;
    gpspp)
        DIM=${DIM:-2048}
        GNN_DEPTH=${GNN_DEPTH:-4}
        BATCH_SIZE=${BATCH_SIZE:-200}
        ;;
    *)
        echo "Error: unknown model '${MODEL}'."
        exit 1
        ;;
esac

TAGS="['${MODEL}','scratch','admet']"

# ── Build dimension flags ────────────────────────────────────────────────────
case "${MODEL}" in
    gcn)   DIM_FLAGS=$(gcn_dim_flags "${DIM}") ;;
    mpnn)  DIM_FLAGS=$(mpnn_dim_flags "${DIM}") ;;
    gpspp) DIM_FLAGS=$(gpspp_dim_flags "${DIM}") ;;
esac

echo "=== Training ${MODEL} from scratch on ADMET (dim=${DIM}, depth=${GNN_DEPTH}) ==="

for task in "${ADMET_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=admet \
        $(wandb_flags "${TAGS}") \
        ++constants.raise_train_error=False \
        ++constants.task=${task} \
        ++datamodule.args.tdc_benchmark_names=${task} \
        ++constants.norm=layer_norm \
        ++architecture.gnn.depth=${GNN_DEPTH} \
        ++datamodule.args.batch_size_training=${BATCH_SIZE} \
        ${DIM_FLAGS} \
        ++architecture.task_heads.${task}.hidden_dims=256 \
        ++architecture.task_heads.${task}.depth=4 \
        ++trainer.model_checkpoint.save_last=False \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
