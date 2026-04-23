#!/usr/bin/env bash
# Run the cell-bioactivity eval for a PairMixer-12M pre-trained on three
# different corpora (ESMC, BBBC047, LitOpenAI-style chain). 6-fold CV per ckpt.
#
# Usage:
#   GPU=5 bash scripts/bioactivity/run_pairmixer_pretrains.sh
#
# Override any checkpoint (path or tag):
#   ESMC_CKPT=/new/path ESMC_TAG=esmc_v3  bash scripts/bioactivity/run_pairmixer_pretrains.sh
#
# Environment:
#   GPU         GPU index to pin to            (default: 0)
#   ESMC_CKPT   ESMC pretrain .ckpt path
#   BBBC_CKPT   BBBC047 pretrain .ckpt path
#   LIT_CKPT    LitOpenAI / lpm24 chain .ckpt path
#   ESMC_TAG / BBBC_TAG / LIT_TAG — labels used in the results CSV's ckpt_tag column.

set -euo pipefail
cd "$(dirname "$0")/../.."

eval "$(conda shell.bash hook)"
conda activate graphium-downstream 2>/dev/null || conda activate graphium

GPU=${GPU:-0}

ESMC_CKPT=${ESMC_CKPT:-models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/2026-04-21_14-03-55_20260421_140355/last.ckpt}
BBBC_CKPT=${BBBC_CKPT:-models_checkpoints/toymix_bbbc047/pairmixer_12M/2026-04-18_11-20-55_20260418_112055/last.ckpt}
LIT_CKPT=${LIT_CKPT:-models_checkpoints/chain/pairmixer_12M_toymix_lpm24_galactica_20260419_170340_20260419_170353/last.ckpt}

ESMC_TAG=${ESMC_TAG:-pairmixer12M_esmc}
BBBC_TAG=${BBBC_TAG:-pairmixer12M_bbbc047}
LIT_TAG=${LIT_TAG:-pairmixer12M_lit_openai}

runs=(
    "${ESMC_TAG}:${ESMC_CKPT}"
    "${BBBC_TAG}:${BBBC_CKPT}"
    "${LIT_TAG}:${LIT_CKPT}"
)

echo "== PairMixer-12M bioactivity sweep on GPU ${GPU} =="
for entry in "${runs[@]}"; do
    tag=${entry%%:*}
    ckpt=${entry##*:}
    echo
    echo "------------------------------------------------------------"
    echo "  tag=${tag}"
    echo "  ckpt=${ckpt}"
    echo "------------------------------------------------------------"
    if [[ ! -f "${ckpt}" ]]; then
        echo "  SKIP: checkpoint not found."
        continue
    fi
    CUDA_VISIBLE_DEVICES=${GPU} python -m downstream.tasks.bioactivity.eval \
        --encoder pairmixer --cv \
        --ckpt "${ckpt}" --ckpt-tag "${tag}" \
        || echo "  WARN: ${tag} failed; continuing"
done

echo
echo "Done. Rows in results/downstream/bioactivity.csv (filter encoder=='pairmixer')."
