#!/usr/bin/env bash
# Pre-train a GNN model on a molecular dataset.
#
# Usage:
#   bash scripts/00_pretrain.sh <model> <dataset> [gpu_id]
#
# Arguments:
#   model    : gcn | mpnn | gpspp | gpspp_800M | gpspp_800M_moe | pairformer
#   dataset  : toymix | largemix | rxrx3 | dti | largemix_rxrx3 | toymix_rxrx3
#              | largemix_dti | toymix_dti | toymix_rxrx3_dti | largemix_rxrx3_dti
#              | rxrx3_dti
#   gpu_id   : CUDA device index (default: 0)
#
# Examples:
#   bash scripts/00_pretrain.sh gpspp largemix 0
#   bash scripts/00_pretrain.sh gcn toymix 1

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> <dataset> [gpu_id]"}
DATASET=${2:?  "Usage: $0 <model> <dataset> [gpu_id]"}
DEVICE=${3:-${DEVICE}}

# Track whether BATCH_SIZE was explicitly set by the user
_USER_BATCH_SIZE=${BATCH_SIZE:-}

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
    gpspp_800M)
        DIM=${DIM:-1536}
        GNN_DEPTH=${GNN_DEPTH:-12}
        BATCH_SIZE=${BATCH_SIZE:-192}
        ;;
    gpspp_800M_moe)
        DIM=${DIM:-1536}
        GNN_DEPTH=${GNN_DEPTH:-12}
        BATCH_SIZE=${BATCH_SIZE:-192}
        ;;
    gpspp_768)
        DIM=${DIM:-768}
        GNN_DEPTH=${GNN_DEPTH:-8}
        BATCH_SIZE=${BATCH_SIZE:-128}
        ;;
    pairformer)
        DIM=${DIM:-256}
        GNN_DEPTH=${GNN_DEPTH:-16}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    pairformer_small)
        DIM=${DIM:-192}
        GNN_DEPTH=${GNN_DEPTH:-8}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    pairformer_medium)
        DIM=${DIM:-256}
        GNN_DEPTH=${GNN_DEPTH:-12}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    pairformer_large)
        DIM=${DIM:-384}
        GNN_DEPTH=${GNN_DEPTH:-16}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    pairformer_boltz)
        DIM=${DIM:-384}
        GNN_DEPTH=${GNN_DEPTH:-48}
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    pairmixer_small|pairmixer_small_mean|pairmixer_small_moe|pairmixer_medium|pairmixer_boltz|pairmixer_boltz_moe)
        ;;  # dims, depth, batch size all in YAML configs
    *)
        echo "Error: unknown model '${MODEL}'."
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
    toymix_rxrx3)
        TASKS=toymix_rxrx3
        TRAINING=toymix_rxrx3
        ARCHITECTURE=toymix
        [[ "${MODEL}" == "gcn" || "${MODEL}" == "mpnn" ]] && BATCH_SIZE=${BATCH_SIZE:-1024}
        ;;
    dti)
        TASKS=dti
        TRAINING=dti
        ARCHITECTURE=largemix
        ;;
    dti_filtered)
        TASKS=dti_filtered
        TRAINING=dti_filtered
        ARCHITECTURE=largemix
        ;;
    largemix_dti)
        TASKS=largemix_dti
        TRAINING=largemix_dti
        ARCHITECTURE=largemix
        ;;
    largemix_dti_filtered)
        TASKS=largemix_dti_filtered
        TRAINING=largemix_dti_filtered
        ARCHITECTURE=largemix
        ;;
    toymix_dti)
        TASKS=toymix_dti
        TRAINING=toymix_dti
        ARCHITECTURE=toymix
        [[ "${MODEL}" == "gcn" || "${MODEL}" == "mpnn" ]] && BATCH_SIZE=${BATCH_SIZE:-1024}
        ;;
    toymix_dti_filtered)
        TASKS=toymix_dti_filtered
        TRAINING=toymix_dti_filtered
        ARCHITECTURE=toymix
        [[ "${MODEL}" == "gcn" || "${MODEL}" == "mpnn" ]] && BATCH_SIZE=${BATCH_SIZE:-1024}
        ;;
    toymix_rxrx3_dti)
        TASKS=toymix_rxrx3_dti
        TRAINING=toymix_rxrx3_dti
        ARCHITECTURE=toymix
        [[ "${MODEL}" == "gcn" || "${MODEL}" == "mpnn" ]] && BATCH_SIZE=${BATCH_SIZE:-1024}
        ;;
    largemix_rxrx3_dti)
        TASKS=largemix_rxrx3_dti
        TRAINING=largemix_rxrx3_dti
        ARCHITECTURE=largemix
        ;;
    rxrx3_dti)
        TASKS=rxrx3_dti
        TRAINING=rxrx3_dti
        ARCHITECTURE=largemix
        ;;
    bbbc047)
        TASKS=bbbc047
        TRAINING=bbbc047
        ARCHITECTURE=largemix
        ;;
    toymix_bbbc047)
        TASKS=toymix_bbbc047
        TRAINING=toymix_bbbc047
        ARCHITECTURE=toymix
        [[ "${MODEL}" == "gcn" || "${MODEL}" == "mpnn" ]] && BATCH_SIZE=${BATCH_SIZE:-1024}
        ;;
    toymix_bbbc047_filtered)
        TASKS=toymix_bbbc047_filtered
        TRAINING=toymix_bbbc047_filtered
        ARCHITECTURE=toymix
        [[ "${MODEL}" == "gcn" || "${MODEL}" == "mpnn" ]] && BATCH_SIZE=${BATCH_SIZE:-1024}
        ;;
    dti_filtered)
        TASKS=dti_filtered
        TRAINING=dti
        ARCHITECTURE=largemix
        ;;
    dti_10k_filtered)
        TASKS=dti_10k_filtered
        TRAINING=dti_10k_filtered
        ARCHITECTURE=largemix
        ;;
    toymix_dti_10k_filtered)
        TASKS=toymix_dti_10k_filtered
        TRAINING=toymix_dti_10k_filtered
        ARCHITECTURE=toymix
        [[ "${MODEL}" == "gcn" || "${MODEL}" == "mpnn" ]] && BATCH_SIZE=${BATCH_SIZE:-1024}
        ;;
    *)
        echo "Error: unknown dataset '${DATASET}'."
        exit 1
        ;;
esac

# Override batch size for toymix with simple models
if [[ "${DATASET}" == "toymix" && ("${MODEL}" == "gcn" || "${MODEL}" == "mpnn") ]]; then
    BATCH_SIZE=${BATCH_SIZE:-1024}
fi

# Cap batch size for DTI datasets (2560-dim output head needs more memory)
# Only applies when BATCH_SIZE was explicitly set; otherwise config handles it.
if [[ -n "${_USER_BATCH_SIZE}" ]]; then
    if [[ "${DATASET}" == "dti" || "${DATASET}" == "dti_filtered" || "${DATASET}" == "dti_10k_filtered" || "${DATASET}" == "toymix_dti_10k_filtered" || "${DATASET}" == "toymix_dti_filtered" || "${DATASET}" == "largemix_dti" || "${DATASET}" == "largemix_dti_filtered" || "${DATASET}" == "toymix_dti" || "${DATASET}" == "toymix_rxrx3_dti" || "${DATASET}" == "rxrx3_dti" || "${DATASET}" == "largemix_rxrx3_dti" ]]; then
        (( BATCH_SIZE > 128 )) && BATCH_SIZE=128
    fi
fi

TAGS="['${MODEL}','pretrain','${DATASET}']"

# ── Build dimension flags ────────────────────────────────────────────────────
case "${MODEL}" in
    gcn)        DIM_FLAGS=$(gcn_dim_flags "${DIM}") ;;
    mpnn)       DIM_FLAGS=$(mpnn_dim_flags "${DIM}") ;;
    gpspp)      DIM_FLAGS=$(gpspp_dim_flags "${DIM}") ;;
    gpspp_800M)     DIM_FLAGS="" ;;  # gpspp_800M sets dims in model config
    gpspp_800M_moe) DIM_FLAGS="" ;;  # gpspp_800M_moe sets dims in model config
    gpspp_768)      DIM_FLAGS="" ;;  # gpspp_768 sets dims in model config
    pairformer)        DIM_FLAGS="" ;;  # pairformer uses config defaults
    pairformer_small)  DIM_FLAGS="" ;;  # dims set in model config
    pairformer_medium) DIM_FLAGS="" ;;  # dims set in model config
    pairformer_large)  DIM_FLAGS="" ;;  # dims set in model config
    pairformer_boltz)  DIM_FLAGS="" ;;  # dims set in model config
    pairmixer_small|pairmixer_small_mean|pairmixer_small_moe|pairmixer_medium|pairmixer_boltz|pairmixer_boltz_moe)  DIM_FLAGS="" ;;
esac

# ── Optional: sample_size for dataset size ablation ──────────────────────────
SAMPLE_FLAGS=""
if [[ -n "${SAMPLE_SIZE:-}" ]]; then
    if [[ "${DATASET}" == "toymix" || "${DATASET}" == "toymix_rxrx3" || "${DATASET}" == "toymix_dti" || "${DATASET}" == "toymix_dti_filtered" || "${DATASET}" == "toymix_dti_10k_filtered" || "${DATASET}" == "toymix_rxrx3_dti" ]]; then
        for t in qm9 tox21 zinc; do
            SAMPLE_FLAGS="${SAMPLE_FLAGS} ++datamodule.args.task_specific_args.${t}.sample_size=${SAMPLE_SIZE}"
        done
    elif [[ "${DATASET}" == "largemix" || "${DATASET}" == "largemix_rxrx3" || "${DATASET}" == "largemix_dti" || "${DATASET}" == "largemix_dti_filtered" || "${DATASET}" == "largemix_rxrx3_dti" ]]; then
        for t in l1000_vcap l1000_mcf7 pcba_1328 pcqm4m_g25 pcqm4m_n4; do
            SAMPLE_FLAGS="${SAMPLE_FLAGS} ++datamodule.args.task_specific_args.${t}.sample_size=${SAMPLE_SIZE}"
        done
    fi
    if [[ "${DATASET}" == "rxrx3" || "${DATASET}" == "largemix_rxrx3" || "${DATASET}" == "toymix_rxrx3" || "${DATASET}" == "toymix_rxrx3_dti" || "${DATASET}" == "rxrx3_dti" || "${DATASET}" == "largemix_rxrx3_dti" ]]; then
        SAMPLE_FLAGS="${SAMPLE_FLAGS} ++datamodule.args.task_specific_args.rxrx3.sample_size=${SAMPLE_SIZE}"
    fi
    if [[ "${DATASET}" == "dti" || "${DATASET}" == "dti_filtered" || "${DATASET}" == "dti_10k_filtered" || "${DATASET}" == "toymix_dti_10k_filtered" || "${DATASET}" == "toymix_dti_filtered" || "${DATASET}" == "largemix_dti" || "${DATASET}" == "largemix_dti_filtered" || "${DATASET}" == "toymix_dti" || "${DATASET}" == "toymix_rxrx3_dti" || "${DATASET}" == "rxrx3_dti" || "${DATASET}" == "largemix_rxrx3_dti" ]]; then
        SAMPLE_FLAGS="${SAMPLE_FLAGS} ++datamodule.args.task_specific_args.dti.sample_size=${SAMPLE_SIZE}"
    fi
fi

# ── Build batch size flag (only override if user explicitly set BATCH_SIZE) ──
BATCH_FLAGS=""
if [[ -n "${_USER_BATCH_SIZE}" ]]; then
    BATCH_FLAGS="++datamodule.args.batch_size_training=${BATCH_SIZE}"
fi

# ── Build depth flags ────────────────────────────────────────────────────────
# Models with dedicated configs set depth internally; don't override via CLI.
DEPTH_FLAGS=""
case "${MODEL}" in
    gpspp_800M|gpspp_800M_moe|gpspp_768|pairformer|pairformer_small|pairformer_medium|pairformer_large|pairformer_boltz|pairmixer_small|pairmixer_small_mean|pairmixer_small_moe|pairmixer_medium|pairmixer_boltz|pairmixer_boltz_moe)
        ;;  # depth comes from model config
    *)
        DEPTH_FLAGS="++architecture.gnn.depth=${GNN_DEPTH}"
        ;;
esac

# ── Run ──────────────────────────────────────────────────────────────────────
echo "=== Pre-training ${MODEL} on ${DATASET} (dim=${DIM:-config}, depth=${GNN_DEPTH:-config}, bs=${BATCH_SIZE:-config}) ==="

CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
    model=${MODEL} \
    accelerator=gpu \
    tasks=${TASKS} \
    training=${TRAINING} \
    architecture=${ARCHITECTURE} \
    $(wandb_flags "${TAGS}") \
    ${DEPTH_FLAGS} \
    ${BATCH_FLAGS} \
    ${DIM_FLAGS} \
    ${SAMPLE_FLAGS} \
    ${EXTRA_FLAGS:-}
