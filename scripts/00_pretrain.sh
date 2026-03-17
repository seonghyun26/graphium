#!/usr/bin/env bash
# Pre-train a GNN model on a molecular dataset.
#
# Usage:
#   bash scripts/00_pretrain.sh <model> <dataset> [gpu_id]
#
# Arguments:
#   model    : gcn | mpnn | gpspp | pairformer
#   dataset  : toymix | largemix | rxrx3 | largemix_rxrx3
#   gpu_id   : CUDA device index (default: 0)
#
# Examples:
#   bash scripts/00_pretrain.sh gpspp largemix 0
#   bash scripts/00_pretrain.sh gcn toymix 1

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> <dataset> [gpu_id]"}
DATASET=${2:?  "Usage: $0 <model> <dataset> [gpu_id]"}
DEVICE=${3:-${DEVICE}}

# ── Model-specific defaults ──────────────────────────────────────────────────
case "${MODEL}" in
    gcn)
        DIM=${DIM:-5120}
        GNN_DEPTH=${GNN_DEPTH:-16}
        BATCH_SIZE=${BATCH_SIZE:-400}
        ;;
    mpnn)
        DIM=${DIM:-900}
        GNN_DEPTH=${GNN_DEPTH:-16}
        BATCH_SIZE=${BATCH_SIZE:-400}
        ;;
    gpspp)
        DIM=${DIM:-1472}
        GNN_DEPTH=${GNN_DEPTH:-8}
        BATCH_SIZE=${BATCH_SIZE:-300}
        ;;
    pairformer)
        DIM=${DIM:-256}
        GNN_DEPTH=${GNN_DEPTH:-16}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    *)
        echo "Error: unknown model '${MODEL}'. Choose from: gcn, mpnn, gpspp, pairformer"
        exit 1
        ;;
esac

# ── Dataset-specific config ──────────────────────────────────────────────────
case "${DATASET}" in
    toymix)
        TASKS=toymix
        TRAINING=toymix
        ARCHITECTURE=toymix
        [[ "${MODEL}" == "gcn" || "${MODEL}" == "mpnn" ]] && BATCH_SIZE=${BATCH_SIZE:-1024}
        ;;
    largemix)
        TASKS=largemix
        TRAINING=largemix
        ARCHITECTURE=largemix
        ;;
    rxrx3)
        TASKS=rxrx3
        TRAINING=rxrx3
        ARCHITECTURE=largemix
        ;;
    largemix_rxrx3)
        TASKS=largemix_rxrx3
        TRAINING=largemix
        ARCHITECTURE=largemix
        ;;
    *)
        echo "Error: unknown dataset '${DATASET}'. Choose from: toymix, largemix, rxrx3, largemix_rxrx3"
        exit 1
        ;;
esac

# Override batch size for toymix with simple models
if [[ "${DATASET}" == "toymix" && ("${MODEL}" == "gcn" || "${MODEL}" == "mpnn") ]]; then
    BATCH_SIZE=${BATCH_SIZE:-1024}
fi

TAGS="['${MODEL}','pretrain','${DATASET}']"

# ── Build dimension flags ────────────────────────────────────────────────────
case "${MODEL}" in
    gcn)        DIM_FLAGS=$(gcn_dim_flags "${DIM}") ;;
    mpnn)       DIM_FLAGS=$(mpnn_dim_flags "${DIM}") ;;
    gpspp)      DIM_FLAGS=$(gpspp_dim_flags "${DIM}") ;;
    pairformer) DIM_FLAGS="" ;;  # pairformer uses config defaults
esac

# ── Optional: sample_size for dataset size ablation ──────────────────────────
SAMPLE_FLAGS=""
if [[ -n "${SAMPLE_SIZE:-}" ]]; then
    if [[ "${DATASET}" == "toymix" ]]; then
        for t in qm9 tox21 zinc; do
            SAMPLE_FLAGS="${SAMPLE_FLAGS} ++datamodule.args.task_specific_args.${t}.sample_size=${SAMPLE_SIZE}"
        done
    elif [[ "${DATASET}" == "largemix" || "${DATASET}" == "largemix_rxrx3" ]]; then
        for t in l1000_vcap l1000_mcf7 pcba_1328 pcqm4m_g25 pcqm4m_n4; do
            SAMPLE_FLAGS="${SAMPLE_FLAGS} ++datamodule.args.task_specific_args.${t}.sample_size=${SAMPLE_SIZE}"
        done
    fi
    if [[ "${DATASET}" == "rxrx3" || "${DATASET}" == "largemix_rxrx3" ]]; then
        SAMPLE_FLAGS="${SAMPLE_FLAGS} ++datamodule.args.task_specific_args.rxrx3.sample_size=${SAMPLE_SIZE}"
    fi
fi

# ── Run ──────────────────────────────────────────────────────────────────────
echo "=== Pre-training ${MODEL} on ${DATASET} (dim=${DIM}, depth=${GNN_DEPTH}, bs=${BATCH_SIZE}) ==="

CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
    model=${MODEL} \
    accelerator=gpu \
    tasks=${TASKS} \
    training=${TRAINING} \
    architecture=${ARCHITECTURE} \
    $(wandb_flags "${TAGS}") \
    ++constants.norm=layer_norm \
    ++architecture.gnn.depth=${GNN_DEPTH} \
    ++datamodule.args.batch_size_training=${BATCH_SIZE} \
    ${DIM_FLAGS} \
    ${SAMPLE_FLAGS} \
    ${EXTRA_FLAGS:-}
