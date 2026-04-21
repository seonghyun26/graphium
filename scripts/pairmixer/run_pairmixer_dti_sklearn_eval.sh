#!/usr/bin/env bash
# Apples-to-apples sklearn DTI eval for a PairMixer checkpoint.
# Mirrors scripts/run_minimol_mole_dti.sh, lets you swap in any PairMixer ckpt.
#
# Usage:
#   bash scripts/pairmixer/run_pairmixer_dti_sklearn_eval.sh <ckpt> [gpu_id] [ckpt_tag]
#
# Env vars (optional):
#   MODEL=pairmixer_12M             : model config name
#   SUBSETS="DAVIS KIBA"            : space-sep subsets
#   METHODS="random cold_target"    : space-sep split methods
#   SEEDS="0 1 2 3 4"               : space-sep seeds
#   HEAD=mlp                        : linear | mlp
#
# Writes per-run rows to results/pairmixer_dti_results.csv. Per-checkpoint
# embeddings are cached at datacache/pairmixer_dti_embeddings/<subset>_<model>_<ckpthash>.pt
# so re-runs across seeds/methods reuse them.

set -euo pipefail
cd "$(dirname "$0")/../.."

eval "$(conda shell.bash hook)"
conda activate graphium

CKPT=${1:?"Usage: $0 <ckpt> [gpu_id] [ckpt_tag]"}
DEVICE=${2:-7}
CKPT_TAG=${3:-$(basename "${CKPT%.ckpt}")}

MODEL=${MODEL:-pairmixer_12M}
SUBSETS=${SUBSETS:-"DAVIS KIBA"}
METHODS=${METHODS:-"random cold_target"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
HEAD=${HEAD:-mlp}

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi
if [[ ! -d "data/dti-eval/splits" ]]; then
    echo "ERROR: data/dti-eval/splits/ missing. Run scripts/data/dti_eval/01_prepare_dti_eval.py first."
    exit 1
fi

echo "=== PairMixer sklearn DTI eval (GPU=${DEVICE}) ==="
echo "    model    : ${MODEL}"
echo "    ckpt     : ${CKPT}"
echo "    ckpt_tag : ${CKPT_TAG}"
echo "    subsets  : ${SUBSETS}"
echo "    methods  : ${METHODS}"
echo "    seeds    : ${SEEDS}"
echo "    head     : ${HEAD}"

FAILED=()

for subset in ${SUBSETS}; do
    for method in ${METHODS}; do
        for seed in ${SEEDS}; do
            echo ""
            echo "--- ${subset}/${method}/seed=${seed} ---"
            CUDA_VISIBLE_DEVICES=${DEVICE} python scripts/pairmixer/pairmixer_dti_eval.py \
                --ckpt "${CKPT}" \
                --model "${MODEL}" \
                --subset "${subset}" \
                --method "${method}" \
                --seed "${seed}" \
                --head "${HEAD}" \
                --ckpt-tag "${CKPT_TAG}" \
                || FAILED+=("${subset}:${method}:seed${seed}")
        done
    done
done

echo ""
echo "================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "=== ALL RUNS SUCCEEDED ==="
else
    echo "=== FAILED (${#FAILED[@]}): ${FAILED[*]} ==="
fi
echo "Results: results/pairmixer_dti_results.csv"
