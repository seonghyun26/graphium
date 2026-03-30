#!/usr/bin/env bash
# Train a model from scratch on ADMET benchmark tasks (no pre-training).
# This serves as the baseline for the dataset ablation study.
#
# Usage:
#   bash scripts/02_scratch_admet.sh <model> [gpu_id]
#
# Arguments:
#   model  : gcn | mpnn | gpspp | gpspp_800M | pairformer | pairformer_boltz
#   gpu_id : CUDA device index (default: 0)
#
# Environment variables (optional):
#   DIM        : hidden dimension (default: model-specific)
#   GNN_DEPTH  : GNN depth (default: model-specific)
#   BATCH_SIZE : batch size (default: model-specific)
#
# Examples:
#   bash scripts/02_scratch_admet.sh gpspp 0
#   bash scripts/02_scratch_admet.sh gpspp_800M 4
#   bash scripts/02_scratch_admet.sh pairformer_boltz 5
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
    gpspp_800M)
        # Uses gpspp_800M config (dim=1536, depth=12); no DIM_FLAGS needed
        BATCH_SIZE=${BATCH_SIZE:-200}
        ;;
    pairformer)
        DIM=${DIM:-256}
        GNN_DEPTH=${GNN_DEPTH:-16}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    pairformer_boltz)
        DIM=${DIM:-384}
        GNN_DEPTH=${GNN_DEPTH:-48}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    pairmixer)
        DIM=${DIM:-256}
        GNN_DEPTH=${GNN_DEPTH:-16}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    pairmixer_small)
        DIM=${DIM:-192}
        GNN_DEPTH=${GNN_DEPTH:-8}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    pairmixer_boltz)
        DIM=${DIM:-384}
        GNN_DEPTH=${GNN_DEPTH:-48}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    *)
        echo "Error: unknown model '${MODEL}'. Choose from: gcn, mpnn, gpspp, gpspp_800M, pairformer, pairformer_boltz, pairmixer, pairmixer_small, pairmixer_boltz"
        exit 1
        ;;
esac

TAGS="['${MODEL}','scratch','admet']"

# ── Build dimension flags ────────────────────────────────────────────────────
case "${MODEL}" in
    gcn)              DIM_FLAGS=$(gcn_dim_flags "${DIM}") ;;
    mpnn)             DIM_FLAGS=$(mpnn_dim_flags "${DIM}") ;;
    gpspp)            DIM_FLAGS=$(gpspp_dim_flags "${DIM}") ;;
    gpspp_800M)       DIM_FLAGS="" ;;  # gpspp_800M sets dims in model config
    pairformer)       DIM_FLAGS="" ;;  # pairformer uses config defaults
    pairformer_boltz) DIM_FLAGS="" ;;
    pairmixer)        DIM_FLAGS="" ;;
    pairmixer_small)  DIM_FLAGS="" ;;
    pairmixer_boltz)  DIM_FLAGS="" ;;  # pairformer_boltz sets dims in model config
esac

# ── Build model-specific extra flags ────────────────────────────────────────
# Models with dedicated configs (gpspp_800M, pairformer_*, pairmixer_*) get
# dims/depth from their YAML — don't override via CLI.
case "${MODEL}" in
    gpspp_800M|pairformer|pairformer_small|pairformer_medium|pairformer_large|pairformer_boltz|pairmixer|pairmixer_small|pairmixer_boltz)
        ARCH_FLAGS="++constants.norm=layer_norm"
        DATAMODULE_FLAGS="++datamodule.args.batch_size_training=${BATCH_SIZE}"
        echo "=== Training ${MODEL} from scratch on ADMET (config defaults) ==="
        ;;
    *)
        ARCH_FLAGS="++constants.norm=layer_norm ++architecture.gnn.depth=${GNN_DEPTH} ${DIM_FLAGS}"
        DATAMODULE_FLAGS="++datamodule.args.batch_size_training=${BATCH_SIZE}"
        echo "=== Training ${MODEL} from scratch on ADMET (dim=${DIM}, depth=${GNN_DEPTH}) ==="
        ;;
esac

# Precision override: pairformer/pairmixer triangle ops overflow fp16
case "${MODEL}" in
    pairformer*|pairmixer*)
        ARCH_FLAGS="${ARCH_FLAGS} ++trainer.trainer.precision=bf16-mixed"
        ;;
esac

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
        ${ARCH_FLAGS} \
        ${DATAMODULE_FLAGS} \
        ++architecture.task_heads.${task}.hidden_dims=256 \
        ++architecture.task_heads.${task}.depth=4 \
        ++trainer.model_checkpoint.save_last=False \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
