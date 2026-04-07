#!/usr/bin/env bash
# Run the full ESM-C DTI embedding pipeline.
#
# Reuses existing data from the ESM-2 pipeline (no TDC re-download needed):
#   - data/dti-scratch/protein-esm2.csv     (protein_id + sequence)
#   - data/dti-scratch/tdcdti-esm2.parquet  (protein-drug SMILES pairings)
#
# Stages:
#   2. Extract ESM-C embeddings (esmc env, Python 3.12)
#   3. Consolidate embeddings   (graphium env)
#   4. Merge with drug SMILES   (graphium env)
#   5. Filter + normalize       (graphium env)
#
# Usage:
#     bash run_esmc_pipeline.sh [GPU_IDS]
#
# Examples:
#     bash run_esmc_pipeline.sh          # single GPU (0)
#     bash run_esmc_pipeline.sh 0,1,2,3  # 4 GPUs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

GPU_IDS="${1:-0}"

# Existing data from the ESM-2 pipeline
PROTEIN_CSV="data/dti-scratch/protein-esm2.csv"
DTI_PARQUET="data/dti-scratch/tdcdti-esm2.parquet"

# Output directories
DTI_INTERMEDIATE_DIR="graphium/data/dti"
DTI_FINAL_DIR="graphium/data/dti-processed"
ESMC_EMBEDDING_DIR="data/protein-esmc"

echo "ESM-C DTI Embedding Pipeline"
echo "========================================"
echo "  Project dir:  ${PROJECT_DIR}"
echo "  GPUs:         ${GPU_IDS}"
echo "  Protein CSV:  ${PROTEIN_CSV}"
echo "  DTI parquet:  ${DTI_PARQUET}"
echo ""

# Verify existing data
if [ ! -f "${PROTEIN_CSV}" ]; then
    echo "ERROR: ${PROTEIN_CSV} not found."
    echo "Run the ESM-2 pipeline first, or provide the protein CSV path."
    exit 1
fi
if [ ! -f "${DTI_PARQUET}" ]; then
    echo "ERROR: ${DTI_PARQUET} not found."
    echo "Run the ESM-2 pipeline first, or provide the DTI parquet path."
    exit 1
fi

# --------------------------------------------------------------------------
# Stage 2: Extract ESM-C 600M embeddings (requires esmc conda env)
# --------------------------------------------------------------------------
echo "[Stage 2] Extracting ESM-C 600M embeddings ..."
echo "  (Running in 'esmc' conda env with Python 3.12)"

# Verify esmc env exists
if ! conda env list 2>/dev/null | grep -q "^esmc "; then
    echo "  ERROR: conda env 'esmc' not found."
    echo "  Run: bash ${SCRIPT_DIR}/setup_esmc_env.sh"
    exit 1
fi

conda run -n esmc python "${SCRIPT_DIR}/02_extract_embeddings_esmc.py" \
    --input "${PROTEIN_CSV}" \
    --output-dir "${ESMC_EMBEDDING_DIR}" \
    --gpus "${GPU_IDS}"
echo ""

# --------------------------------------------------------------------------
# Stage 3: Consolidate per-protein embeddings
# --------------------------------------------------------------------------
echo "[Stage 3] Consolidating ESM-C embeddings ..."
python "${SCRIPT_DIR}/03_consolidate_embeddings_esmc.py" \
    --embedding-dir "${ESMC_EMBEDDING_DIR}/embedding" \
    --protein-csv "${PROTEIN_CSV}" \
    --output-dir "${DTI_INTERMEDIATE_DIR}"
echo ""

# --------------------------------------------------------------------------
# Stage 4: Merge embeddings with drug SMILES
# --------------------------------------------------------------------------
echo "[Stage 4] Merging embeddings with drug SMILES ..."
python "${SCRIPT_DIR}/04_merge_datasets_esmc.py" \
    --embeddings "${DTI_INTERMEDIATE_DIR}/protein-esmc.csv" \
    --dti "${DTI_PARQUET}" \
    --output "${DTI_INTERMEDIATE_DIR}/tdcdti-esmc.parquet"
echo ""

# --------------------------------------------------------------------------
# Stage 5: Filter to 100K + normalize
# --------------------------------------------------------------------------
echo "[Stage 5] Filtering to 100K rows + normalizing ..."
python "${SCRIPT_DIR}/05_filter_normalize_esmc.py" \
    --input "${DTI_INTERMEDIATE_DIR}/tdcdti-esmc.parquet" \
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
ls -lh "${DTI_FINAL_DIR}"/dti_esmc_* 2>/dev/null || echo "  (no files yet)"
echo ""
echo "Use in training with:  tasks=dti_esmc"
