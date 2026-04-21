#!/usr/bin/env bash
# Orchestrate the GRAM-DTI binary-classification benchmark across all encoders,
# subsets, methods, and folds.
#
# Prerequisites (already produced by stages under scripts/data/dti_classif_eval/):
#   data/dti-classif-eval/<subset>.parquet                              # 4 files
#   data/dti-classif-eval/splits/<subset>_<method>_fold<i>.pt           # 90 files
#   data/dti-classif-eval/protein-esmc.parquet                          # ESM-C cache
#
# Usage:
#   bash scripts/gram_dti_eval/run_all.sh <pairmixer_ckpt>              # full sweep, all 3 encoders
#   ENCODERS=minimol,mole bash scripts/gram_dti_eval/run_all.sh _       # baselines only
#
# Env overrides:
#   ENCODERS=pairmixer,minimol,mole          # which eval scripts to run
#   SUBSETS=yamanishi_08,hetionet,activation,inhibition
#   METHODS=warm,drug_cold,target_cold
#   FOLDS=                                   # space-sep ints; empty = all folds (10 DTI / 5 MoA)
#   HEAD=logreg                              # logreg | mlp
#   GPU=0                                    # for mole / pairmixer extraction
#   PARQUET_DIR=data/dti-classif-eval

set -euo pipefail

PAIRMIXER_CKPT=${1:-_}
ENCODERS=${ENCODERS:-pairmixer,minimol,mole}
SUBSETS=${SUBSETS:-yamanishi_08,hetionet,activation,inhibition}
METHODS=${METHODS:-warm,drug_cold,target_cold}
FOLDS=${FOLDS:-}
HEAD=${HEAD:-logreg}
GPU=${GPU:-0}
PARQUET_DIR=${PARQUET_DIR:-data/dti-classif-eval}

PYTHON=${PYTHON:-/home/shpark/.conda/envs/graphium/bin/python}

if [[ ! -d "${PARQUET_DIR}/splits" ]]; then
    echo "ERROR: ${PARQUET_DIR}/splits not found."
    echo "       Run scripts/data/dti_classif_eval/{01_collect_dtiam_proteins,02_prepare_dti_classif}.py first."
    exit 1
fi

IFS=',' read -ra SUBSET_LIST  <<< "${SUBSETS}"
IFS=',' read -ra METHOD_LIST  <<< "${METHODS}"
IFS=',' read -ra ENCODER_LIST <<< "${ENCODERS}"

# Convert FOLDS env (space-sep) to argparse positional flags.
fold_args=()
if [[ -n "${FOLDS}" ]]; then
    fold_args=(--folds ${FOLDS})
fi

common_args=(
    --subsets ${SUBSET_LIST[@]}
    --methods ${METHOD_LIST[@]}
    --head ${HEAD}
    --parquet-dir ${PARQUET_DIR}
    "${fold_args[@]}"
)

for encoder in "${ENCODER_LIST[@]}"; do
    echo
    echo "============================================================"
    echo "  encoder: ${encoder}"
    echo "============================================================"
    case "${encoder}" in
        minimol)
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/gram_dti_eval/eval_minimol.py "${common_args[@]}"
            ;;
        mole)
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/gram_dti_eval/eval_mole.py "${common_args[@]}" --device cuda:0
            ;;
        pairmixer)
            if [[ "${PAIRMIXER_CKPT}" == "_" || ! -f "${PAIRMIXER_CKPT}" ]]; then
                echo "  SKIP: pairmixer requires a checkpoint as the first argument."
                continue
            fi
            CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/gram_dti_eval/eval_pairmixer.py \
                --ckpt "${PAIRMIXER_CKPT}" "${common_args[@]}" --device cuda:0
            ;;
        *)
            echo "  ERROR: unknown encoder '${encoder}' (expected: pairmixer|minimol|mole)"
            exit 1
            ;;
    esac
done

echo
echo "Done. Per-encoder rows in results/gram_dti_eval/<encoder>_results.csv."
