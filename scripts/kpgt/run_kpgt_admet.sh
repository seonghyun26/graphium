#!/usr/bin/env bash
# Evaluate KPGT (figshare base.pth) on TDC ADMET (22 tasks) and Polaris ADME-Fang (6 tasks).
#
# Usage:
#   bash scripts/kpgt/run_kpgt_admet.sh [gpu_id]
#
# Env vars:
#   BENCHMARK      : tdc | polaris | both (default: both)
#   HEAD           : linear | mlp        (default: mlp)
#   SEEDS          : space-separated     (default: "0 1 2")
#   TASKS_OVERRIDE : space-separated, overrides both task lists
#
# Embedding extraction runs in a sibling ``kpgt`` conda env via subprocess
# (the graphium env can't host KPGT — DGL 0.7 vs torch 2.7 ABI mismatch).
# The eval script itself runs in the graphium env (sklearn, TDC, Polaris).
#
# Per-SMILES embeddings live in ``datacache/kpgt_embeddings/`` and are reused
# across seeds; the kpgt env only fires for the first seed of each task.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

eval "$(conda shell.bash hook)"
conda activate graphium

DEVICE="${1:-0}"
BENCHMARK="${BENCHMARK:-both}"
HEAD="${HEAD:-mlp}"
SEEDS="${SEEDS:-0 1 2}"
KPGT_DEVICE="${KPGT_DEVICE:-cpu}"

# Sanity check the figshare ckpt is in place. The encoder also enforces this,
# but failing fast before the first run beats failing on every task.
KPGT_CKPT="${KPGT_CKPT:-downloads/kpgt/base.pth}"
if [[ ! -f "${KPGT_CKPT}" ]]; then
    cat <<EOF >&2
[error] ${KPGT_CKPT} missing.
        Run scripts/kpgt/download_kpgt_baseline.sh first, or follow the
        manual instructions printed by that script (figshare anonymous-share
        downloads can fail and need a browser fallback).
EOF
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
    echo "=== [${bench}] ${task} seed=${seed} ==="
    CUDA_VISIBLE_DEVICES="${DEVICE}" \
        python scripts/kpgt/kpgt_eval.py \
        --benchmark "${bench}" \
        --task "${task}" \
        --seed "${seed}" \
        --head "${HEAD}" \
        --device "${KPGT_DEVICE}" \
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
echo "Results: results/kpgt_results.csv"
