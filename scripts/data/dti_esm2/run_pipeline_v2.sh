#!/usr/bin/env bash
# Run the DTI v2 pipeline: train-split-only DTI data with ESM-2 embeddings.
#
# Reuses existing protein ESM-2 embeddings from the v1 pipeline.
# Only re-collects DTI pairs (train split only) and re-runs merge + filter.
#
# Stages:
#   1. Collect DTI data from TDC (train split only)
#   2-3. SKIPPED: reuses existing protein ESM-2 embeddings
#   4. Merge protein embeddings with drug SMILES
#   5. Filter to 100K rows + normalize → dti_esm2_100k_v2.*
#
# Prerequisites:
#   - Protein embeddings from v1 pipeline: data/dti-scratch/protein-esm2.csv
#
# Usage:
#     bash run_pipeline_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

# Directories
DTI_V1_DIR="data/dti-scratch"           # v1 intermediates (protein embeddings)
DTI_V2_DIR="data/dti-scratch-v2"        # v2 intermediates
DTI_FINAL_DIR="data/dti-processed"      # final training-ready files

# Protein embeddings from v1 (must already exist)
PROTEIN_EMB="${DTI_V1_DIR}/protein-esm2.csv"
if [ ! -f "${PROTEIN_EMB}" ]; then
    echo "ERROR: Protein embeddings not found at ${PROTEIN_EMB}"
    echo "       Run the v1 pipeline first: bash run_pipeline.sh"
    exit 1
fi

echo "DTI v2 Pipeline (train-split-only)"
echo "========================================"
echo "  Project dir:  ${PROJECT_DIR}"
echo "  Reusing:      ${PROTEIN_EMB}"
echo ""

mkdir -p "${DTI_V2_DIR}"

# --------------------------------------------------------------------------
# Stage 1: Collect DTI data from TDC (TRAIN SPLIT ONLY)
# --------------------------------------------------------------------------
PROTEIN_CSV="${DTI_V2_DIR}/protein-drug-trainonly.csv"
if [ -f "${PROTEIN_CSV}" ]; then
    echo "[Stage 1] Skipping (${PROTEIN_CSV} already exists)"
else
    echo "[Stage 1] Collecting DTI data from TDC (train split only) ..."
    python "${SCRIPT_DIR}/01_collect_tdc_data.py" \
        --output "${PROTEIN_CSV}" \
        --cache-dir "data/tdc_cache" \
        --train-only
fi
echo ""

# --------------------------------------------------------------------------
# Stages 2-3: SKIPPED (reusing existing protein ESM-2 embeddings)
# --------------------------------------------------------------------------
echo "[Stages 2-3] Skipping (reusing protein embeddings from v1)"
echo ""

# --------------------------------------------------------------------------
# Stage 4: Merge embeddings with drug SMILES
# --------------------------------------------------------------------------
MERGED="${DTI_V2_DIR}/tdcdti-esm2-v2.parquet"
if [ -f "${MERGED}" ]; then
    echo "[Stage 4] Skipping (${MERGED} already exists)"
else
    echo "[Stage 4] Merging embeddings with train-only drug SMILES ..."
    python "${SCRIPT_DIR}/04_merge_datasets.py" \
        --embeddings "${PROTEIN_EMB}" \
        --dti "${PROTEIN_CSV}" \
        --output "${MERGED}"
fi
echo ""

# --------------------------------------------------------------------------
# Stage 5: Filter to 100K + normalize
# --------------------------------------------------------------------------
echo "[Stage 5] Filtering to 100K rows + normalizing ..."
python "${SCRIPT_DIR}/05_filter_normalize.py" \
    --input "${MERGED}" \
    --output-dir "${DTI_V2_DIR}" \
    --final-dir "${DTI_FINAL_DIR}" \
    --final-prefix "dti_esm2_100k_v2" \
    --seed 42 \
    --target-size 100000
echo ""

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo "========================================"
echo "Pipeline complete!"
echo ""
echo "Final files in ${DTI_FINAL_DIR}/:"
ls -lh "${DTI_FINAL_DIR}"/dti_esm2_*v2* 2>/dev/null || echo "  (no files yet)"
echo ""
echo "Use in training with:  tasks=dti_v2  or  tasks=toymix_dti_v2"
