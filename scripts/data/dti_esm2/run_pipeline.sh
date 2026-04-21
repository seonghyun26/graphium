#!/usr/bin/env bash
# Run the full ESM-2 DTI embedding pipeline.
#
# Downloads DTI data from TDC, extracts ESM-2 3B protein embeddings,
# and builds the final dataset for Graphium pre-training.
#
# Stages:
#   1. Collect DTI data from TDC (DAVIS, KIBA, BindingDB)
#   2. Extract ESM-2 3B embeddings (2560-dim) from protein sequences
#   3. Consolidate per-protein embeddings into single file
#   4. Merge protein embeddings with drug SMILES
#   5. Filter to 100K rows + normalize
#
# Usage:
#     bash run_pipeline.sh [GPU_IDS]
#
# Examples:
#     bash run_pipeline.sh          # single GPU (0)
#     bash run_pipeline.sh 0,1      # 2 GPUs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

GPU_IDS="${1:-0}"

# Directories
DTI_RAW_DIR="data/dti-scratch"
DTI_INTERMEDIATE_DIR="data/dti"
DTI_FINAL_DIR="data/dti-processed"
ESM2_EMBEDDING_DIR="data/protein-esm2"

echo "ESM-2 DTI Embedding Pipeline"
echo "========================================"
echo "  Project dir:  ${PROJECT_DIR}"
echo "  GPUs:         ${GPU_IDS}"
echo ""

# --------------------------------------------------------------------------
# Stage 1: Collect DTI data from TDC
# --------------------------------------------------------------------------
PROTEIN_CSV="${DTI_RAW_DIR}/protein-drug.csv"
if [ -f "${PROTEIN_CSV}" ]; then
    echo "[Stage 1] Skipping (${PROTEIN_CSV} already exists)"
else
    echo "[Stage 1] Collecting DTI data from TDC ..."
    mkdir -p "${DTI_RAW_DIR}"
    python "${SCRIPT_DIR}/01_collect_tdc_data.py" \
        --output "${PROTEIN_CSV}" \
        --cache-dir "${DTI_RAW_DIR}/tdc_cache"
fi
echo ""

# --------------------------------------------------------------------------
# Stage 2: Extract ESM-2 3B embeddings from protein sequences
# --------------------------------------------------------------------------
echo "[Stage 2] Extracting ESM-2 3B embeddings ..."
python "${SCRIPT_DIR}/02_extract_embeddings.py" \
    --input "${PROTEIN_CSV}" \
    --output-dir "${ESM2_EMBEDDING_DIR}" \
    --gpus "${GPU_IDS}"
echo ""

# --------------------------------------------------------------------------
# Stage 3: Consolidate per-protein embeddings
# --------------------------------------------------------------------------
echo "[Stage 3] Consolidating ESM-2 embeddings ..."
python "${SCRIPT_DIR}/03_consolidate_embeddings.py" \
    --embedding-dir "${ESM2_EMBEDDING_DIR}" \
    --protein-csv "${PROTEIN_CSV}" \
    --output-dir "${DTI_RAW_DIR}"
echo ""

# --------------------------------------------------------------------------
# Stage 4: Merge embeddings with drug SMILES
# --------------------------------------------------------------------------
echo "[Stage 4] Merging embeddings with drug SMILES ..."
python "${SCRIPT_DIR}/04_merge_datasets.py" \
    --embeddings "${DTI_RAW_DIR}/protein-esm2.csv" \
    --dti "${PROTEIN_CSV}" \
    --output "${DTI_RAW_DIR}/tdcdti-esm2.parquet"
echo ""

# --------------------------------------------------------------------------
# Stage 5: Filter to 100K + normalize
# --------------------------------------------------------------------------
echo "[Stage 5] Filtering to 100K rows + normalizing ..."
python "${SCRIPT_DIR}/05_filter_normalize.py" \
    --input "${DTI_RAW_DIR}/tdcdti-esm2.parquet" \
    --output-dir "${DTI_INTERMEDIATE_DIR}" \
    --final-dir "${DTI_FINAL_DIR}" \
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
ls -lh "${DTI_FINAL_DIR}"/dti_esm2_* 2>/dev/null || echo "  (no files yet)"
echo ""
echo "Use in training with:  tasks=dti  or  tasks=toymix_dti"
