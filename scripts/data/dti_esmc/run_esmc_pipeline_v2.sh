#!/usr/bin/env bash
# Run the DTI v2 ESM-C pipeline: train-split-only DTI data with ESM-C embeddings.
#
# Fully independent — does not require the ESM-2 v2 pipeline.
# Reuses existing protein ESM-C embeddings from the v1 pipeline.
#
# Stages:
#   1. Collect DTI data from TDC (train split only)
#   2-3. SKIPPED: reuses existing protein ESM-C embeddings
#   4. Merge ESM-C embeddings with train-only drug SMILES
#   5. Filter to 100K rows + normalize → dti_esmc_100k_v2.*
#
# Prerequisites:
#   - ESM-C consolidated embeddings: graphium/data/dti/protein-esmc.csv
#
# Usage:
#     bash run_esmc_pipeline_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESM2_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/../dti_esm2" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

# Existing data
ESMC_EMB="graphium/data/dti/protein-esmc.csv"

# Output directories (shared with ESM-2 v2 for stage 1)
DTI_V2_DIR="data/dti-scratch-v2"
DTI_FINAL_DIR="data/dti-processed"

# Verify prerequisites
if [ ! -f "${ESMC_EMB}" ]; then
    echo "ERROR: ESM-C embeddings not found at ${ESMC_EMB}"
    echo "       Run the v1 ESM-C pipeline first: bash run_esmc_pipeline.sh"
    exit 1
fi

echo "DTI v2 ESM-C Pipeline (train-split-only)"
echo "========================================"
echo "  Project dir:  ${PROJECT_DIR}"
echo "  ESM-C emb:    ${ESMC_EMB}"
echo ""

mkdir -p "${DTI_V2_DIR}"

# --------------------------------------------------------------------------
# Stage 1: Collect DTI data from TDC (TRAIN SPLIT ONLY)
# Shared with ESM-2 v2 pipeline — same output file
# --------------------------------------------------------------------------
PROTEIN_CSV="${DTI_V2_DIR}/protein-drug-trainonly.csv"
if [ -f "${PROTEIN_CSV}" ]; then
    echo "[Stage 1] Skipping (${PROTEIN_CSV} already exists)"
else
    echo "[Stage 1] Collecting DTI data from TDC (train split only) ..."
    python "${ESM2_SCRIPT_DIR}/01_collect_tdc_data.py" \
        --output "${PROTEIN_CSV}" \
        --cache-dir "data/tdc_cache" \
        --train-only
fi
echo ""

# --------------------------------------------------------------------------
# Stages 2-3: SKIPPED (reusing existing protein ESM-C embeddings)
# --------------------------------------------------------------------------
echo "[Stages 2-3] Skipping (reusing ESM-C embeddings from v1)"
echo ""

# --------------------------------------------------------------------------
# Stage 4: Merge ESM-C embeddings with train-only drug SMILES
# --------------------------------------------------------------------------
MERGED="${DTI_V2_DIR}/tdcdti-esmc-v2.parquet"
if [ -f "${MERGED}" ]; then
    echo "[Stage 4] Skipping (${MERGED} already exists)"
else
    echo "[Stage 4] Merging ESM-C embeddings with train-only drug SMILES ..."
    python "${SCRIPT_DIR}/04_merge_datasets_esmc.py" \
        --embeddings "${ESMC_EMB}" \
        --dti "${PROTEIN_CSV}" \
        --output "${MERGED}"
fi
echo ""

# --------------------------------------------------------------------------
# Stage 5: Filter to 100K + normalize
# --------------------------------------------------------------------------
echo "[Stage 5] Filtering to 100K rows + normalizing ..."
python "${SCRIPT_DIR}/05_filter_normalize_esmc.py" \
    --input "${MERGED}" \
    --output-dir "${DTI_V2_DIR}" \
    --final-dir "${DTI_FINAL_DIR}" \
    --final-prefix "dti_esmc_100k_v2" \
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
ls -lh "${DTI_FINAL_DIR}"/dti_esmc_*v2* 2>/dev/null || echo "  (no files yet)"
echo ""
echo "Use in training with:  tasks=dti_esmc_v2"
