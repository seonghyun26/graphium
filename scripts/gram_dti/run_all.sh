#!/usr/bin/env bash
# Run the GRAM-DTI eval across all encoders × subsets × methods × folds.
#
# Usage:
#   bash scripts/gram_dti/run_all.sh                      # minimol + mole (no ckpt needed)
#   bash scripts/gram_dti/run_all.sh <pairmixer_ckpt>     # add pairmixer
#
# Env overrides:
#   ENCODERS=pairmixer,minimol,mole            # which encoders to run
#   HEAD=mlp                                    # mlp | autogluon | linear
#   SUBSETS=yamanishi_08,hetionet,activation,inhibition
#   METHODS=warm,drug_cold,target_cold
#   FOLDS=                                      # space-sep ints; empty = all
#   GPU=0                                       # for pairmixer/mole extraction
#
# Prerequisites:
#   data/downstream/gram_dti/*.parquet          (bash scripts/gram_dti/prepare_data.sh ...)
#
# Results: results/downstream/gram_dti.csv (one row per encoder × subset × method × fold × head).

set -euo pipefail
cd "$(dirname "$0")/../.."

eval "$(conda shell.bash hook)"
conda activate graphium-downstream 2>/dev/null || conda activate graphium

PAIRMIXER_CKPT=${1:-_}
ENCODERS=${ENCODERS:-minimol,mole}
HEAD=${HEAD:-mlp}
SUBSETS=${SUBSETS:-yamanishi_08,hetionet,activation,inhibition}
METHODS=${METHODS:-warm,drug_cold,target_cold}
FOLDS=${FOLDS:-}
GPU=${GPU:-0}

if [[ "${PAIRMIXER_CKPT}" != "_" && ",${ENCODERS}," != *",pairmixer,"* ]]; then
    ENCODERS="pairmixer,${ENCODERS}"
fi

if [[ ! -d "../data/downstream/gram_dti" ]]; then
    echo "ERROR: ../data/downstream/gram_dti/ missing. Run prepare_data.sh first."
    exit 1
fi

IFS=',' read -ra SUBSET_LIST  <<< "${SUBSETS}"
IFS=',' read -ra METHOD_LIST  <<< "${METHODS}"
IFS=',' read -ra ENCODER_LIST <<< "${ENCODERS}"

FOLD_ARGS=()
if [[ -n "${FOLDS}" ]]; then
    FOLD_ARGS=(--folds ${FOLDS})
fi

for encoder in "${ENCODER_LIST[@]}"; do
    echo
    echo "============================================================"
    echo "  encoder: ${encoder}   head: ${HEAD}"
    echo "============================================================"
    case "${encoder}" in
        pairmixer)
            case "${PAIRMIXER_CKPT}" in
                _) echo "  SKIP: pairmixer requires a .ckpt path or 'scratch' as the first argument."; continue ;;
                scratch|random|none) ;;
                *) [[ -f "${PAIRMIXER_CKPT}" ]] || { echo "  SKIP: pairmixer ckpt not found: ${PAIRMIXER_CKPT}"; continue; } ;;
            esac
            CUDA_VISIBLE_DEVICES=${GPU} python -m downstream.tasks.gram_dti.eval \
                --encoder pairmixer --ckpt "${PAIRMIXER_CKPT}" --head "${HEAD}" \
                --subsets "${SUBSET_LIST[@]}" --methods "${METHOD_LIST[@]}" \
                "${FOLD_ARGS[@]}"
            ;;
        minimol)
            CUDA_VISIBLE_DEVICES=${GPU} python -m downstream.tasks.gram_dti.eval \
                --encoder minimol --head "${HEAD}" \
                --subsets "${SUBSET_LIST[@]}" --methods "${METHOD_LIST[@]}" \
                "${FOLD_ARGS[@]}"
            ;;
        mole)
            CUDA_VISIBLE_DEVICES=${GPU} python -m downstream.tasks.gram_dti.eval \
                --encoder mole --head "${HEAD}" \
                --subsets "${SUBSET_LIST[@]}" --methods "${METHOD_LIST[@]}" \
                "${FOLD_ARGS[@]}"
            ;;
        *)
            echo "ERROR: unknown encoder '${encoder}'"
            exit 1
            ;;
    esac
done

echo
echo "Done. Rows in results/downstream/gram_dti.csv."
