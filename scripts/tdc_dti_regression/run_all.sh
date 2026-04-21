#!/usr/bin/env bash
# Run TDC DTI regression eval across encoders × subsets × methods × seeds.
#
# Usage:
#   bash scripts/tdc_dti_regression/run_all.sh                  # minimol + mole
#   bash scripts/tdc_dti_regression/run_all.sh <pairmixer_ckpt> # add pairmixer
#
# Env overrides:
#   ENCODERS=pairmixer,minimol,mole
#   HEAD=mlp                                # mlp | autogluon | linear
#   SUBSETS=DAVIS,KIBA                       # space OR comma sep
#   METHODS=random,cold_target
#   SEEDS=0,1,2,3,4
#   GPU=0
#
# Results: results/downstream/tdc_dti_regression.csv

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate graphium

PAIRMIXER_CKPT=${1:-_}
ENCODERS=${ENCODERS:-minimol,mole}
HEAD=${HEAD:-mlp}
SUBSETS=${SUBSETS:-DAVIS,KIBA}
METHODS=${METHODS:-random,cold_target}
SEEDS=${SEEDS:-0,1,2,3,4}
GPU=${GPU:-0}

if [[ "${PAIRMIXER_CKPT}" != "_" ]]; then
    ENCODERS="pairmixer,${ENCODERS}"
fi

if [[ ! -d "data/downstream/tdc_dti_regression" ]]; then
    echo "ERROR: data/downstream/tdc_dti_regression/ missing. Run prepare_data.sh first."
    exit 1
fi

IFS=',' read -ra SUBSET_LIST  <<< "${SUBSETS}"
IFS=',' read -ra METHOD_LIST  <<< "${METHODS}"
IFS=',' read -ra SEED_LIST    <<< "${SEEDS}"
IFS=',' read -ra ENCODER_LIST <<< "${ENCODERS}"

for encoder in "${ENCODER_LIST[@]}"; do
    echo
    echo "============================================================"
    echo "  encoder: ${encoder}   head: ${HEAD}"
    echo "============================================================"
    case "${encoder}" in
        pairmixer)
            if [[ "${PAIRMIXER_CKPT}" == "_" || ! -f "${PAIRMIXER_CKPT}" ]]; then
                echo "  SKIP: pairmixer requires a .ckpt as the first argument."
                continue
            fi
            CUDA_VISIBLE_DEVICES=${GPU} python -m downstream.tasks.tdc_dti_regression.eval \
                --encoder pairmixer --ckpt "${PAIRMIXER_CKPT}" --head "${HEAD}" \
                --subsets "${SUBSET_LIST[@]}" --methods "${METHOD_LIST[@]}" \
                --seeds "${SEED_LIST[@]}"
            ;;
        minimol|mole)
            CUDA_VISIBLE_DEVICES=${GPU} python -m downstream.tasks.tdc_dti_regression.eval \
                --encoder "${encoder}" --head "${HEAD}" \
                --subsets "${SUBSET_LIST[@]}" --methods "${METHOD_LIST[@]}" \
                --seeds "${SEED_LIST[@]}"
            ;;
        *)
            echo "ERROR: unknown encoder '${encoder}'"
            exit 1
            ;;
    esac
done

echo
echo "Done. Rows in results/downstream/tdc_dti_regression.csv."
