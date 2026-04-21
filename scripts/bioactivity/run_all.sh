#!/usr/bin/env bash
# Run the bioactivity eval across all encoders. One row per (encoder, fold)
# appended to results/downstream/bioactivity.csv.
#
# Usage:
#   bash scripts/bioactivity/run_all.sh                    # ecfp + minimol + mole + cpcnn
#   bash scripts/bioactivity/run_all.sh <pairmixer_ckpt>   # add pairmixer
#
# Env overrides:
#   ENCODERS=ecfp,minimol,mole,cpcnn,pairmixer
#   GPU=0
#   CPCNN_CSV=/home/shpark/prj-molrepr/datacache/jump_cpcnn/jump_cpcnn_smiles_embeddings.csv
#   SINGLE=1                              # skip --cv, run single-split only
#   EXTRA_FLAGS="--epochs 50 --seed 7"    # passed to every encoder

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate graphium-downstream 2>/dev/null || conda activate graphium

PAIRMIXER_CKPT=${1:-_}
ENCODERS=${ENCODERS:-ecfp,minimol,mole,cpcnn}
GPU=${GPU:-0}
CPCNN_CSV=${CPCNN_CSV:-/home/shpark/prj-molrepr/datacache/jump_cpcnn/jump_cpcnn_smiles_embeddings.csv}
SINGLE=${SINGLE:-0}
EXTRA_FLAGS=${EXTRA_FLAGS:-}

if [[ "${PAIRMIXER_CKPT}" != "_" ]]; then
    ENCODERS="pairmixer,${ENCODERS}"
fi

if [[ ! -f "../data/downstream/bioactivity/cell_bioactivity.csv" ]]; then
    echo "ERROR: ../data/downstream/bioactivity/cell_bioactivity.csv missing."
    echo "       Run: bash scripts/bioactivity/prepare_data.sh <chembl_33.db>"
    exit 1
fi

CV_FLAG=(--cv)
if [[ "${SINGLE}" == "1" ]]; then
    CV_FLAG=()
fi

IFS=',' read -ra ENCODER_LIST <<< "${ENCODERS}"

for encoder in "${ENCODER_LIST[@]}"; do
    echo
    echo "============================================================"
    echo "  encoder: ${encoder}"
    echo "============================================================"
    case "${encoder}" in
        pairmixer)
            if [[ "${PAIRMIXER_CKPT}" == "_" || ! -f "${PAIRMIXER_CKPT}" ]]; then
                echo "  SKIP: pairmixer requires a .ckpt as the first argument."
                continue
            fi
            CUDA_VISIBLE_DEVICES=${GPU} python -m downstream.tasks.bioactivity.eval \
                --encoder pairmixer --ckpt "${PAIRMIXER_CKPT}" "${CV_FLAG[@]}" ${EXTRA_FLAGS}
            ;;
        cpcnn)
            if [[ ! -f "${CPCNN_CSV}" ]]; then
                echo "  SKIP: CPCNN CSV not found at ${CPCNN_CSV}"
                continue
            fi
            CUDA_VISIBLE_DEVICES=${GPU} python -m downstream.tasks.bioactivity.eval \
                --encoder cpcnn --cpcnn-csv "${CPCNN_CSV}" "${CV_FLAG[@]}" ${EXTRA_FLAGS}
            ;;
        ecfp|minimol|mole)
            CUDA_VISIBLE_DEVICES=${GPU} python -m downstream.tasks.bioactivity.eval \
                --encoder "${encoder}" "${CV_FLAG[@]}" ${EXTRA_FLAGS}
            ;;
        *)
            echo "ERROR: unknown encoder '${encoder}'"
            exit 1
            ;;
    esac
done

echo
echo "Done. Rows in results/downstream/bioactivity.csv."
