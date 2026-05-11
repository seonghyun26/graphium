#!/usr/bin/env bash
# Evaluate a PairMixer + MPNN++ fingerprint ensemble on the 5 representative
# ADMET tasks (one per ADMET category).
#
# Usage:
#   CKPT_A=.../pairmixer_12M/last.ckpt \
#   CKPT_B=.../mpnnpp_12M/last.ckpt \
#   bash scripts/pairmixer/run_ensemble_admet.sh [gpu_id]
#
# Optional env vars:
#   MODEL_A        : model config name for ckpt-a  (default: pairmixer_12M)
#   MODEL_B        : model config name for ckpt-b  (default: mpnnpp_12M)
#   CKPT_TAG_A     : short label for model A in CSV (default: ckpt stem)
#   CKPT_TAG_B     : short label for model B in CSV (default: ckpt stem)
#   HEAD           : linear | mlp  (default: mlp)
#   SEEDS          : space-separated seeds  (default: "0 1 2")
#   TASKS_OVERRIDE : override the 5-task list (space-separated)
#   BENCHMARK      : tdc | polaris (default: tdc)
#   MODEL_DEVICE   : torch device string (default: cuda:0)
#   EMBED_BATCH_SIZE : backbone forward batch size (default: 32)
#
# Results → results/ensemble_admet_results.csv

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-graphium}"

DEVICE="${1:-0}"
MODEL_A="${MODEL_A:-pairmixer_12M}"
MODEL_B="${MODEL_B:-mpnnpp_12M}"
HEAD="${HEAD:-mlp}"
SEEDS="${SEEDS:-0 1 2}"
BENCHMARK="${BENCHMARK:-tdc}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda:0}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"

if [[ -z "${CKPT_A:-}" ]]; then
    echo "ERROR: CKPT_A env var required (path to model-A .ckpt)." >&2; exit 1
fi
if [[ -z "${CKPT_B:-}" ]]; then
    echo "ERROR: CKPT_B env var required (path to model-B .ckpt)." >&2; exit 1
fi
if [[ ! -f "${CKPT_A}" ]]; then
    echo "ERROR: CKPT_A not found: ${CKPT_A}" >&2; exit 1
fi
if [[ ! -f "${CKPT_B}" ]]; then
    echo "ERROR: CKPT_B not found: ${CKPT_B}" >&2; exit 1
fi

# 5 representative ADMET tasks (one per category — same as ablation scripts)
TASKS=(
    lipophilicity_astrazeneca   # Absorption (regression)
    bbb_martins                 # Distribution (classification)
    cyp3a4_veith                # Metabolism (classification)
    half_life_obach             # Excretion (regression)
    ld50_zhu                    # Toxicity (regression)
)
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
    read -ra TASKS <<< "${TASKS_OVERRIDE}"
fi

echo "============================================================"
echo "  ADMET ENSEMBLE: ${MODEL_A} + ${MODEL_B}"
echo "  CKPT_A: ${CKPT_A}"
echo "  CKPT_B: ${CKPT_B}"
echo "  Tasks: ${#TASKS[@]}  Seeds: [${SEEDS}]  Head: ${HEAD}"
echo "  GPU: ${DEVICE}  benchmark: ${BENCHMARK}"
echo "============================================================"

FAILED=()

for task in "${TASKS[@]}"; do
    for seed in ${SEEDS}; do
        echo ""
        echo "=== ${task}  seed=${seed} ==="
        CUDA_VISIBLE_DEVICES="${DEVICE}" \
            python scripts/pairmixer/ensemble_admet_eval.py \
            --ckpt-a  "${CKPT_A}" \
            --ckpt-b  "${CKPT_B}" \
            --model-a "${MODEL_A}" \
            --model-b "${MODEL_B}" \
            --benchmark "${BENCHMARK}" \
            --task "${task}" \
            --seed "${seed}" \
            --head "${HEAD}" \
            --device "${MODEL_DEVICE}" \
            --embed-batch-size "${EMBED_BATCH_SIZE}" \
            ${CKPT_TAG_A:+--ckpt-tag-a "${CKPT_TAG_A}"} \
            ${CKPT_TAG_B:+--ckpt-tag-b "${CKPT_TAG_B}"} \
            || FAILED+=("${task}:seed${seed}")
    done
done

echo ""
echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "  ALL RUNS SUCCEEDED"
else
    echo "  FAILED (${#FAILED[@]}): ${FAILED[*]}"
fi
echo "  Results: results/ensemble_admet_results.csv"
echo "============================================================"
