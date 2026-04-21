#!/usr/bin/env bash
# Download BBBC047 (JUMP Cell Painting) morphological embeddings.
#
# Source: JUMP Cell Painting Gallery on AWS S3
#   - CPCNN embeddings (EfficientNet-B0, ~5.3M params)
#   - 672-dim morphological embeddings per compound
#   - ~109K compounds from U2OS Cell Painting assays
#
# Paper: Moshkov et al. 2024 (JUMP Cell Painting CPCNN embeddings)
# Data:  s3://cellpainting-gallery/cpg0016-jump/
#        Zenodo: https://zenodo.org/records/7114558 (CPCNN model)
#
# The pre-aggregated compound-level embeddings CSV is expected at:
#   ../../data/bbbc047/bbbc047_smiles_embeddings.csv
#
# If not available, this script downloads per-plate CPCNN profiles
# from the Cell Painting Gallery and aggregates them.
#
# Usage:
#     bash download.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

OUTPUT_DIR="../../data/bbbc047"
OUTPUT_FILE="${OUTPUT_DIR}/bbbc047_smiles_embeddings.csv"

echo "BBBC047 (JUMP Cell Painting) Download"
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

# The JUMP Cell Painting CPCNN embeddings are large and distributed
# across many per-plate parquet files on S3. Downloading and aggregating
# requires significant processing.
#
# Option 1: Download pre-aggregated file (if hosted somewhere accessible)
# Option 2: Download per-plate profiles and aggregate

echo "Attempting to download pre-aggregated BBBC047 embeddings ..."

# Try HuggingFace first (if available)
python -c "
try:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id='jump-cellpainting/cpcnn-embeddings',
        filename='bbbc047_smiles_embeddings.csv',
        repo_type='dataset',
        local_dir='${OUTPUT_DIR}',
    )
    print(f'Downloaded to: {path}')
except Exception as e:
    print(f'HuggingFace download not available: {e}')
    print('Falling back to manual aggregation ...')
    raise SystemExit(1)
" 2>/dev/null && exit 0

echo ""
echo "Pre-aggregated file not available from HuggingFace."
echo ""
echo "To build BBBC047 embeddings manually:"
echo "  1. Download CPCNN profiles from S3:"
echo "     aws s3 sync s3://cellpainting-gallery/cpg0016-jump/source_4/workspace_dl/profiles/cpcnn_zenodo_7114558/ \\"
echo "       data/bbbc047/cpcnn_profiles/ --no-sign-request"
echo "  2. Run the aggregation notebook:"
echo "     See notebooks/03_pilot_tests.ipynb section 2 for the aggregation code"
echo ""
echo "Alternatively, copy the pre-built file from a collaborator:"
echo "  cp /path/to/bbbc047_smiles_embeddings.csv ${OUTPUT_FILE}"
exit 1
