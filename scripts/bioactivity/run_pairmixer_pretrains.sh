#!/usr/bin/env bash
# Run the cell-bioactivity eval for a PairMixer-12M under three pretraining
# regimes (ESMC, BBBC047, Scratch/toymix-only). 6-fold CV per ckpt.
#
# "Scratch" here = PairMixer-12M trained from random init directly on the
# cell-bioactivity labels (no prior multi-task pretrain). Heads-up: because the
# encoder has already seen every fold's labels during its own training, the
# 6-fold CV numbers include train/test leakage — useful as a ceiling / sanity
# check, not as a clean downstream number.
#
# Usage:
#   GPU=5 bash scripts/bioactivity/run_pairmixer_pretrains.sh
#
# Override any checkpoint (path or tag):
#   ESMC_CKPT=/new/path ESMC_TAG=esmc_v3  bash scripts/bioactivity/run_pairmixer_pretrains.sh
#
# Environment:
#   GPU           GPU index to pin to          (default: 0)
#   ESMC_CKPT     ESMC pretrain .ckpt path
#   BBBC_CKPT     BBBC047 pretrain .ckpt path
#   SCRATCH_CKPT  random-init-trained-on-bioactivity .ckpt path
#   ESMC_TAG / BBBC_TAG / SCRATCH_TAG — labels used in the results CSV.

set -euo pipefail
cd "$(dirname "$0")/../.."

eval "$(conda shell.bash hook)"
conda activate graphium-downstream 2>/dev/null || conda activate graphium

GPU=${GPU:-0}

ESMC_CKPT=${ESMC_CKPT:-models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/2026-04-21_14-03-55_20260421_140355/last.ckpt}
BBBC_CKPT=${BBBC_CKPT:-models_checkpoints/toymix_bbbc047/pairmixer_12M/2026-04-18_11-20-55_20260418_112055/last.ckpt}
SCRATCH_CKPT=${SCRATCH_CKPT:-models_checkpoints/cell_bioactivity_scratch/pairmixer_12M_20260420_184815/scratch_cell_bioactivity_pairmixer_12M_epochepoch=039_20260420_184815.ckpt}

ESMC_TAG=${ESMC_TAG:-pairmixer12M_esmc}
BBBC_TAG=${BBBC_TAG:-pairmixer12M_bbbc047}
SCRATCH_TAG=${SCRATCH_TAG:-pairmixer12M_scratch}

runs=(
    "${ESMC_TAG}:${ESMC_CKPT}"
    "${BBBC_TAG}:${BBBC_CKPT}"
    "${SCRATCH_TAG}:${SCRATCH_CKPT}"
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
        --featurize-n-jobs 1 \
        || echo "  WARN: ${tag} failed; continuing"
done

echo
echo "Done. Rows in results/downstream/bioactivity.csv (filter encoder=='pairmixer')."
