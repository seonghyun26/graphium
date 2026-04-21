#!/usr/bin/env bash
# Run the full LPM-24 + text-encoder embedding pipeline.
#
# Downloads L+M-24 (ACL 2024) from HuggingFace, extracts text-encoder
# embeddings from molecule captions, and builds the final dataset
# for Graphium pre-training.
#
# Stages:
#   1. Download LPM-24 from HuggingFace
#   2. Extract embeddings from captions with the chosen text encoder
#   3. Build final dataset (Z-score normalize + CSV)
#
# Supported encoders (all <= 1024-d, single-GPU-friendly):
#   pubmedbert   microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext  (768-d)
#   galactica    facebook/galactica-125m                                        (768-d)
#   biolinkbert  michiyasunaga/BioLinkBERT-large                                (1024-d)
#
# Usage:
#     bash run_lpm24_pipeline.sh [--model MODEL] [--gpu GPU_ID]
#
# Examples:
#     bash run_lpm24_pipeline.sh                                  # pubmedbert on GPU 1
#     bash run_lpm24_pipeline.sh --model galactica                # galactica on GPU 1
#     bash run_lpm24_pipeline.sh --model biolinkbert --gpu 2      # biolinkbert on GPU 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"

# Defaults.  GPU 1 is the free slot on this server; override with --gpu if needed.
MODEL="pubmedbert"
GPU_ID="1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --gpu)
            GPU_ID="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash run_lpm24_pipeline.sh [--model MODEL] [--gpu GPU_ID]" >&2
            exit 1 ;;
    esac
done

case "${MODEL}" in
    pubmedbert|galactica|biolinkbert) ;;
    *) echo "ERROR: --model must be one of pubmedbert|galactica|biolinkbert (got '${MODEL}')" >&2; exit 1 ;;
esac

# Directories
LPM24_DIR="data/lpm24"
FINAL_DIR="data/dti-processed"

echo "LPM-24 + Text-Encoder Embedding Pipeline"
echo "========================================"
echo "  Project dir: ${PROJECT_DIR}"
echo "  Model:       ${MODEL}"
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
# Stage 2: Extract text-encoder embeddings
# --------------------------------------------------------------------------
echo "[Stage 2] Extracting ${MODEL} embeddings ..."
python "${SCRIPT_DIR}/02_extract_embeddings.py" \
    --model "${MODEL}" \
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
    --model "${MODEL}" \
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
ls -lh "${FINAL_DIR}/lpm24_${MODEL}"* 2>/dev/null || echo "  (no files yet)"
echo ""
echo "Use in training with:  tasks=lpm24  or  tasks=toymix_lpm24"
echo "  (point df_path at data/dti-processed/lpm24_${MODEL}.csv)"
