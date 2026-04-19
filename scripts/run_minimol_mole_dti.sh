#!/usr/bin/env bash
# Evaluate MiniMol and MolE on TDC DTI subsets sequentially, pinned to one GPU.
#
# Assumes data/dti-eval/<subset>.csv and data/dti-eval/splits/*.pt already exist
# (run scripts/data/dti_eval/01_prepare_dti_eval.py first).
#
# Usage:
#   bash scripts/run_minimol_mole_dti.sh [gpu_id]
#
# Env vars (all optional):
#   MODELS     : minimol, mole, or "minimol mole"   (default: "minimol mole")
#   SUBSETS    : space-sep subsets                   (default: "DAVIS KIBA")
#   METHODS    : space-sep split methods             (default: "random cold_target")
#   SEEDS      : space-sep seeds                     (default: "0 1 2 3 4")
#   HEAD       : linear | mlp                        (default: mlp)
#
# Writes per-run rows to:
#   results/minimol_dti_results.csv
#   results/mole_dti_results.csv

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate graphium

DEVICE=${1:-5}
MODELS=${MODELS:-"minimol mole"}
SUBSETS=${SUBSETS:-"DAVIS KIBA"}
METHODS=${METHODS:-"random cold_target"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
HEAD=${HEAD:-mlp}

if [[ ! -d "data/dti-eval/splits" ]]; then
    echo "ERROR: data/dti-eval/splits/ missing. Run scripts/data/dti_eval/01_prepare_dti_eval.py first."
    exit 1
fi

echo "=== MiniMol + MolE DTI eval (GPU=${DEVICE}) ==="
echo "    models  : ${MODELS}"
echo "    subsets : ${SUBSETS}"
echo "    methods : ${METHODS}"
echo "    seeds   : ${SEEDS}"
echo "    head    : ${HEAD}"

FAILED=()

run_one() {
    local model=$1 subset=$2 method=$3 seed=$4
    local script
    case "${model}" in
        minimol) script="scripts/minimol/minimol_dti_eval.py" ;;
        mole)    script="scripts/mole/mole_dti_eval.py" ;;
        *) echo "ERROR: unknown model '${model}'"; return 1 ;;
    esac
    echo ""
    echo "--- [${model}] ${subset}/${method}/seed=${seed} ---"
    CUDA_VISIBLE_DEVICES=${DEVICE} python "${script}" \
        --subset "${subset}" --method "${method}" --seed "${seed}" --head "${HEAD}" \
        || FAILED+=("${model}:${subset}:${method}:seed${seed}")
}

for model in ${MODELS}; do
    for subset in ${SUBSETS}; do
        for method in ${METHODS}; do
            for seed in ${SEEDS}; do
                run_one "${model}" "${subset}" "${method}" "${seed}"
            done
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
echo "Results:"
echo "  results/minimol_dti_results.csv"
echo "  results/mole_dti_results.csv"
