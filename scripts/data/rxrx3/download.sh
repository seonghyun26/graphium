#!/usr/bin/env bash
# Download RxRx3 cell morphology embeddings.
#
# Source: Recursion Pharmaceuticals RxRx3 dataset
#   - Pre-computed OpenPhenom-S/16 embeddings (ViT-S/16, 22M params)
#   - 384-dim morphological embeddings per compound
#   - ~17K compounds from cell painting assays
#
# Paper: "Masked Autoencoders for Microscopy are Scalable Learners
#         of Cellular Biology" (CVPR 2024, arXiv 2404.10242)
# Model: https://huggingface.co/recursionpharma/OpenPhenom
# Data:  https://www.rxrx.ai/rxrx3
#
# Output: data/rxrx3/rxrx3_smiles_embeddings.csv
#
# Usage:
#     bash download.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

OUTPUT_DIR="data/rxrx3"
OUTPUT_FILE="${OUTPUT_DIR}/rxrx3_smiles_embeddings.csv"

echo "RxRx3 Download"
echo "========================================"
echo "  Project dir: ${PROJECT_DIR}"
echo "  Output:      ${OUTPUT_FILE}"
echo ""

if [ -f "${OUTPUT_FILE}" ]; then
    echo "Already exists: ${OUTPUT_FILE}"
    wc -l "${OUTPUT_FILE}" | awk '{print "  Rows: " $1-1}'
    echo "  Skipping download."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

echo "Downloading RxRx3 embeddings from HuggingFace ..."
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='recursionpharma/rxrx3',
    filename='rxrx3_smiles_embeddings.csv',
    repo_type='dataset',
    local_dir='${OUTPUT_DIR}',
)
print(f'Downloaded to: {path}')
"

if [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo "Success!"
    wc -l "${OUTPUT_FILE}" | awk '{print "  Rows: " $1-1}'
    head -1 "${OUTPUT_FILE}" | tr ',' '\n' | grep -c feature_ | xargs -I{} echo "  Embedding dims: {}"
else
    echo "ERROR: Download failed. File not found at ${OUTPUT_FILE}"
    echo ""
    echo "Manual download options:"
    echo "  1. Visit https://www.rxrx.ai/rxrx3 and download the embeddings CSV"
    echo "  2. pip install huggingface_hub && huggingface-cli download recursionpharma/rxrx3 rxrx3_smiles_embeddings.csv --repo-type dataset --local-dir ${OUTPUT_DIR}"
    exit 1
fi
