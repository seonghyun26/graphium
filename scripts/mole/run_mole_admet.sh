#!/usr/bin/env bash
# Evaluate MolE embeddings on TDC ADMET (22 tasks) and Polaris ADME-Fang (6 tasks).
#
# Usage:
#   bash scripts/mole/run_mole_admet.sh [gpu_id]
#
# Env vars:
#   BENCHMARK      : tdc | polaris | both (default: both)
#   HEAD           : linear | mlp (default: mlp)
#   SEEDS          : space-separated seeds (default: "0 1 2")
#   TASKS_OVERRIDE : override the task list (space-separated)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

eval "$(conda shell.bash hook)"
conda activate graphium

DEVICE="${1:-7}"
BENCHMARK="${BENCHMARK:-both}"
HEAD="${HEAD:-mlp}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda:0}"

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
    echo "=== [${bench}] ${task} seed=${seed} ==="
    CUDA_VISIBLE_DEVICES="${DEVICE}" \
        python scripts/mole/mole_eval.py \
        --benchmark "${bench}" \
        --task "${task}" \
        --seed "${seed}" \
        --head "${HEAD}" \
        --device "${MODEL_DEVICE}" \
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
echo "Results: results/mole_results.csv"
