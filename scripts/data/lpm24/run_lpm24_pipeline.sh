#!/usr/bin/env bash
# Run the full LPM-24 + PubMedBERT embedding pipeline.
#
# Downloads L+M-24 (ACL 2024) from HuggingFace, extracts PubMedBERT
# embeddings from molecule captions, and builds the final dataset
# for Graphium pre-training.
#
# Stages:
#   1. Download LPM-24 from HuggingFace
#   2. Extract PubMedBERT embeddings from captions
#   3. Build final dataset (Z-score normalize + CSV)
#
# Usage:
#     bash run_lpm24_pipeline.sh [GPU_ID]
#
# Examples:
#     bash run_lpm24_pipeline.sh        # single GPU (0)
#     bash run_lpm24_pipeline.sh 2      # GPU 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

GPU_ID="${1:-0}"

# Directories
LPM24_DIR="data/lpm24"
FINAL_DIR="data/dti-processed"

echo "LPM-24 + PubMedBERT Embedding Pipeline"
echo "========================================"
echo "  Project dir: ${PROJECT_DIR}"
echo "  GPU:         ${GPU_ID}"
echo ""

# --------------------------------------------------------------------------
# Stage 1: Download LPM-24 from HuggingFace
# --------------------------------------------------------------------------
echo "[Stage 1] Downloading LPM-24 dataset ..."
python "${SCRIPT_DIR}/01_download_lpm24.py" \
    --output-dir "${LPM24_DIR}"
echo ""

# --------------------------------------------------------------------------
# Stage 2: Extract PubMedBERT embeddings
# --------------------------------------------------------------------------
echo "[Stage 2] Extracting PubMedBERT embeddings ..."
python "${SCRIPT_DIR}/02_extract_embeddings_pubmedbert.py" \
    --input "${LPM24_DIR}/lpm24_raw.parquet" \
    --output-dir "${LPM24_DIR}" \
    --gpu "${GPU_ID}" \
    --text-field caption
echo ""

# --------------------------------------------------------------------------
# Stage 3: Build final dataset (normalize + CSV)
# --------------------------------------------------------------------------
echo "[Stage 3] Building final dataset ..."
python "${SCRIPT_DIR}/03_build_lpm24_dataset.py" \
    --input "${LPM24_DIR}/lpm24_embeddings.pt" \
    --final-dir "${FINAL_DIR}" \
    --seed 42
echo ""

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo "========================================"
echo "Pipeline complete!"
echo ""
echo "Final files in ${FINAL_DIR}/:"
ls -lh "${FINAL_DIR}"/lpm24_* 2>/dev/null || echo "  (no files yet)"
echo ""
echo "Use in training with:  tasks=lpm24  or  tasks=toymix_lpm24"
