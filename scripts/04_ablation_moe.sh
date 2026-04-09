#!/usr/bin/env bash
# ============================================================================
# ABLATION STUDY: Mixture-of-Experts on PairMixer pair-track transition
# ============================================================================
# Runs baseline vs MoE on two GPUs in parallel, then compares results.
#
# Default: pairmixer_10M on toymix (fast; ~minutes per run).
# Override MODEL_BASE for larger variants (e.g., pairmixer_10M).
#
# Usage:
#   bash scripts/04_ablation_moe.sh [gpu_baseline] [gpu_moe]
#
# Environment variables:
#   MODEL_BASE   : base model (default: pairmixer_10M)
#   DATASET      : dataset (default: toymix)
#   MOE_EXPERTS  : number of experts (default: 4)
#   MOE_TOP_K    : top-k routing (default: 2)
#   SEEDS        : space-separated seeds (default: "0 1 2")
# ============================================================================

source "$(dirname "$0")/common.sh"

GPU_BASE=${1:-0}
GPU_MOE=${2:-1}

MODEL_BASE=${MODEL_BASE:-pairmixer_10M}
MODEL_MOE=${MODEL_BASE}_moe
DATASET=${DATASET:-toymix}
MOE_EXPERTS=${MOE_EXPERTS:-4}
MOE_TOP_K=${MOE_TOP_K:-2}
MOE_COEFF=${MOE_COEFF:-0.01}
SEEDS=(${SEEDS:-0 1 2})

echo "============================================================"
echo "  MoE ABLATION: ${MODEL_BASE} vs ${MODEL_MOE}"
echo "  Dataset: ${DATASET}   Seeds: ${SEEDS[*]}"
echo "  MoE config: ${MOE_EXPERTS} experts, top-${MOE_TOP_K}"
echo "  GPUs: baseline=${GPU_BASE}, moe=${GPU_MOE}"
echo "============================================================"

for SEED_VAL in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Seed ${SEED_VAL}: launching baseline (GPU ${GPU_BASE}) and MoE (GPU ${GPU_MOE}) in parallel <<<"

    # Baseline
    SEED=${SEED_VAL} \
        bash "$(dirname "$0")/00_pretrain.sh" "${MODEL_BASE}" "${DATASET}" "${GPU_BASE}" &
    PID_BASE=$!

    # MoE
    SEED=${SEED_VAL} \
        bash "$(dirname "$0")/00_pretrain.sh" "${MODEL_MOE}" "${DATASET}" "${GPU_MOE}" &
    PID_MOE=$!

    # Wait for both
    echo "    Waiting for baseline (PID ${PID_BASE}) and MoE (PID ${PID_MOE})..."
    wait ${PID_BASE}
    RC_BASE=$?
    wait ${PID_MOE}
    RC_MOE=$?

    echo "    Seed ${SEED_VAL} done: baseline=${RC_BASE}, moe=${RC_MOE}"
    if [[ ${RC_BASE} -ne 0 ]]; then
        echo "    ERROR: baseline failed (exit ${RC_BASE})"
    fi
    if [[ ${RC_MOE} -ne 0 ]]; then
        echo "    ERROR: MoE failed (exit ${RC_MOE})"
    fi
done

echo ""
echo "============================================================"
echo "  MoE ABLATION COMPLETE"
echo "  Results in: ${RESULTS_DIR}/experiment_results.csv"
echo "  Filter by model: ${MODEL_BASE}, ${MODEL_MOE}"
echo "============================================================"
