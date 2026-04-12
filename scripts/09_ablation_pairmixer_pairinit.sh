#!/usr/bin/env bash
# ============================================================================
# ABLATION: Pair Representation Initialization (PairMixer 20M on toymix)
# ============================================================================
# Compares pair_init variants against the OPM-only baseline.
#
# Conditions (all use pairmixer_20M backbone: D_s=256, D_z=192, depth=20):
#   baseline        — OPM only (model=pairmixer_20M)                 — usually already running
#   structural      — OPM + graph_distance + adjacency
#   struct_ef       — structural + 1-hop edge_feat projection
#   struct_path     — structural + Graphormer-style path_edge
#   full            — structural + edge_feat + path_edge (all sources)
#
# Background
# ----------
# The baseline (``pairmixer_20M``, the default) is typically started
# separately and should NOT be re-run unless ``INCLUDE_BASELINE=1``.
# This script runs the four *new* arms in parallel, one arm per GPU,
# for each seed.  All runs are tagged ``ablation_pairinit`` so the
# results are easy to filter in
# ``results/experiment_results.csv``.
#
# Usage
# -----
#   bash scripts/09_ablation_pairmixer_pairinit.sh [gpu_struct] [gpu_ef] [gpu_path] [gpu_full] [gpu_baseline]
#
# Environment variables
# ---------------------
#   SEEDS             : space-separated seeds (default: "0 1 2")
#   INCLUDE_BASELINE  : set to 1 to also re-run the baseline (uses gpu_baseline)
#   EXTRA_FLAGS       : additional Hydra CLI flags (appended to every run)
#   SKIP_ARMS         : space-separated labels to skip (e.g. SKIP_ARMS="struct_ef full")
#
# Results
# -------
# Filter ``results/experiment_results.csv`` (or W&B) by ``wandb_tags``
# containing ``ablation_pairinit``.  Compare the five arms by the label
# tag: ``baseline``, ``structural``, ``struct_ef``, ``struct_path``, ``full``.
# ============================================================================

set -u
source "$(dirname "$0")/common.sh"

GPU_STRUCT=${1:-0}
GPU_EF=${2:-1}
GPU_PATH=${3:-2}
GPU_FULL=${4:-3}
GPU_BASELINE=${5:-4}

SEEDS=(${SEEDS:-0 1 2})
DATASET=toymix
ABLATION_TAG=ablation_pairinit
SKIP_ARMS=${SKIP_ARMS:-}

# ── Build the condition table: model:label:gpu ──────────────────────────────
declare -a CONDITIONS=()

if [[ "${INCLUDE_BASELINE:-0}" == "1" ]]; then
    CONDITIONS+=("pairmixer_20M:baseline:${GPU_BASELINE}")
fi
CONDITIONS+=(
    "pairmixer_20M_pairinit_structural:structural:${GPU_STRUCT}"
    "pairmixer_20M_pairinit_struct_ef:struct_ef:${GPU_EF}"
    "pairmixer_20M_pairinit_struct_path:struct_path:${GPU_PATH}"
    "pairmixer_20M_pairinit_full:full:${GPU_FULL}"
)

# Honor SKIP_ARMS: drop any condition whose label is in the skip list.
if [[ -n "${SKIP_ARMS}" ]]; then
    filtered=()
    for cond in "${CONDITIONS[@]}"; do
        IFS=':' read -r _m label _g <<< "${cond}"
        skip=0
        for s in ${SKIP_ARMS}; do
            [[ "${s}" == "${label}" ]] && { skip=1; break; }
        done
        (( skip == 0 )) && filtered+=("${cond}")
    done
    CONDITIONS=("${filtered[@]}")
fi

echo "============================================================"
echo "  PAIR-INIT ABLATION: PairMixer 20M (${DATASET})"
echo "  Seeds: ${SEEDS[*]}"
echo "  Conditions:"
for cond in "${CONDITIONS[@]}"; do
    IFS=':' read -r model label gpu <<< "${cond}"
    echo "    [GPU ${gpu}] ${model} (${label})"
done
[[ -n "${SKIP_ARMS}" ]] && echo "  Skipped arms: ${SKIP_ARMS}"
echo "============================================================"

# ── Pre-training: all arms of each seed run in parallel ─────────────────────
for SEED_VAL in "${SEEDS[@]}"; do
    echo ""
    echo "=== Seed ${SEED_VAL}: launching ${#CONDITIONS[@]} arm(s) in parallel ==="
    declare -a PIDS=()
    declare -a LABELS=()

    for cond in "${CONDITIONS[@]}"; do
        IFS=':' read -r MODEL LABEL GPU <<< "${cond}"
        TAGS="['${MODEL}','pretrain','${DATASET}','${ABLATION_TAG}','${LABEL}']"

        echo "  [GPU ${GPU}] seed=${SEED_VAL} ${MODEL} (${LABEL})"

        # Pass EXTRA_FLAGS as an environment variable (MUST precede the
        # command to be exported to the child shell).  Seed and tags go
        # via Hydra CLI overrides — the training/model YAML's default
        # seed of 42 is overridden here.
        EXTRA_FLAGS="++constants.seed=${SEED_VAL} ++constants.wandb.tags=${TAGS} ${EXTRA_FLAGS:-}" \
            bash "$(dirname "$0")/00_pretrain.sh" "${MODEL}" "${DATASET}" "${GPU}" \
            > "logs/ablation_pairinit_${LABEL}_seed${SEED_VAL}.log" 2>&1 &

        PIDS+=($!)
        LABELS+=("${LABEL}")
    done

    # Wait for every arm of this seed before starting the next seed.
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[i]}
        LABEL=${LABELS[i]}
        if wait "${PID}"; then
            echo "  [OK] seed=${SEED_VAL} ${LABEL}"
        else
            echo "  [FAIL] seed=${SEED_VAL} ${LABEL} (exit $?)"
        fi
    done

    unset PIDS LABELS
done

echo ""
echo "============================================================"
echo "  PAIR-INIT ABLATION COMPLETE"
echo "  Results: ${RESULTS_DIR}/experiment_results.csv"
echo "  Filter by wandb_tags containing '${ABLATION_TAG}'"
echo "  Compare labels: baseline (pre-existing), structural, struct_ef, struct_path, full"
echo "============================================================"
