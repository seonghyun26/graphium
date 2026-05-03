#!/usr/bin/env bash
# Evaluate PairMixer embeddings (frozen backbone) on TDC ADMET (22 tasks) and
# Polaris ADME-Fang (6 tasks). Mirrors scripts/mole/run_mole_admet.sh — same
# benchmark loop, same seeds, same sklearn head — swapping MolE for PairMixer.
#
# Usage:
#   bash scripts/pairmixer/run_pairmixer_admet.sh [gpu_id]
#
# Required env vars:
#   CKPT          : path to the PairMixer .ckpt (pretrained backbone).
#
# Optional env vars:
#   MODEL         : model config name (default: pairmixer_12M).
#   CKPT_TAG      : short label for CSV rows (default: ckpt filename stem).
#   BENCHMARK     : tdc | polaris | both (default: both).
#   HEAD          : linear | mlp (default: mlp).
#   SEEDS         : space-separated seeds (default: "0 1 2").
#   TASKS_OVERRIDE: override the task list (space-separated).
#   MODEL_DEVICE  : torch device string passed to the eval (default: cuda:0).
#   EMBED_BATCH_SIZE     : batch size for the backbone forward pass (default: 32).
#   FEATURIZE_N_JOBS     : joblib workers for graphium featurization (default: 8).
#
# Writes per-run rows to results/pairmixer_admet_results.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-graphium}"

DEVICE="${1:-7}"
MODEL="${MODEL:-pairmixer_12M}"
BENCHMARK="${BENCHMARK:-both}"
HEAD="${HEAD:-mlp}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda:0}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"
FEATURIZE_N_JOBS="${FEATURIZE_N_JOBS:-8}"

if [[ -z "${CKPT:-}" ]]; then
    echo "ERROR: CKPT env var is required (path to PairMixer .ckpt)." >&2
    exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

TDC_TASKS=("${ADMET_TASKS[@]}")
POLARIS_TASKS=("${POLARIS_ADME_TASKS[@]}")

if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
    read -ra OVERRIDE_TASKS <<< "${TASKS_OVERRIDE}"
    TDC_TASKS=("${OVERRIDE_TASKS[@]}")
    POLARIS_TASKS=("${OVERRIDE_TASKS[@]}")
fi

FAILED=()

run_one() {
    local bench="$1" task="$2" seed="$3"
    echo ""
    echo "=== [${bench}] ${task} seed=${seed} (ckpt=${CKPT_TAG:-$(basename "${CKPT}" .ckpt)}) ==="
    CUDA_VISIBLE_DEVICES="${DEVICE}" \
        python scripts/pairmixer/pairmixer_admet_eval.py \
        --ckpt "${CKPT}" \
        --model "${MODEL}" \
        --benchmark "${bench}" \
        --task "${task}" \
        --seed "${seed}" \
        --head "${HEAD}" \
        --device "${MODEL_DEVICE}" \
        --embed-batch-size "${EMBED_BATCH_SIZE}" \
        --featurize-n-jobs "${FEATURIZE_N_JOBS}" \
        ${CKPT_TAG:+--ckpt-tag "${CKPT_TAG}"} \
        || FAILED+=("${bench}:${task}:seed${seed}")
}

if [[ "${BENCHMARK}" == "tdc" || "${BENCHMARK}" == "both" ]]; then
    for task in "${TDC_TASKS[@]}"; do
        for seed in ${SEEDS}; do
            run_one tdc "${task}" "${seed}"
        done
    done
fi

if [[ "${BENCHMARK}" == "polaris" || "${BENCHMARK}" == "both" ]]; then
    for task in "${POLARIS_TASKS[@]}"; do
        for seed in ${SEEDS}; do
            run_one polaris "${task}" "${seed}"
        done
    done
fi

echo ""
echo "================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "=== ALL RUNS SUCCEEDED ==="
else
    echo "=== FAILED (${#FAILED[@]}): ${FAILED[*]} ==="
fi
echo "Results: results/pairmixer_admet_results.csv"
