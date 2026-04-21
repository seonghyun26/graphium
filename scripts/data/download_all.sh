#!/usr/bin/env bash
# Download and prepare all pre-training datasets.
#
# Skips datasets that already exist. Pass GPU_IDS for embedding extraction.
#
# Usage:
#     bash download_all.sh [GPU_IDS]
#
# Examples:
#     bash download_all.sh          # single GPU (0)
#     bash download_all.sh 0,1      # 2 GPUs for embedding extraction

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

GPU_IDS="${1:-0}"

echo "=========================================="
echo "  Graphium Pre-training Data Pipeline"
echo "=========================================="
echo "  Project dir: ${PROJECT_DIR}"
echo "  GPUs:        ${GPU_IDS}"
echo ""

# --------------------------------------------------------------------------
# 1. Molecular graph datasets (ToyMix + LargeMix)
# --------------------------------------------------------------------------
echo "--- [1/6] ToyMix + LargeMix ---"
if [ -d "data/graphium/neurips2023/small-dataset" ] && [ -d "data/graphium/neurips2023/large-dataset" ]; then
    echo "  Already exists. Skipping."
else
    bash "${SCRIPT_DIR}/download_graphium_dataset.sh"
fi
echo ""

# --------------------------------------------------------------------------
# 2. RxRx3 cell morphology embeddings
# --------------------------------------------------------------------------
echo "--- [2/6] RxRx3 ---"
bash "${SCRIPT_DIR}/rxrx3/download.sh"
echo ""

# --------------------------------------------------------------------------
# 3. BBBC047 cell morphology embeddings
# --------------------------------------------------------------------------
echo "--- [3/6] BBBC047 ---"
bash "${SCRIPT_DIR}/bbbc047/download.sh"
echo ""

# --------------------------------------------------------------------------
# 4. DTI + ESM-2 protein embeddings
# --------------------------------------------------------------------------
echo "--- [4/6] DTI (ESM-2) ---"
if [ -f "data/dti-processed/dti_esm2_100k.csv" ]; then
    echo "  Already exists. Skipping."
else
    bash "${SCRIPT_DIR}/dti_esm2/run_pipeline.sh" "${GPU_IDS}"
fi
echo ""

# --------------------------------------------------------------------------
# 5. DTI + ESM-C protein embeddings
# --------------------------------------------------------------------------
echo "--- [5/6] DTI (ESM-C) ---"
if [ -f "data/dti-processed/dti_esmc_100k.csv" ]; then
    echo "  Already exists. Skipping."
else
    bash "${SCRIPT_DIR}/dti_esmc/run_pipeline.sh" "${GPU_IDS}"
fi
echo ""

# --------------------------------------------------------------------------
# 6. LPM-24 language embeddings
# --------------------------------------------------------------------------
echo "--- [6/6] LPM-24 ---"
if [ -f "data/dti-processed/lpm24_pubmedbert.csv" ]; then
    echo "  Already exists. Skipping."
else
    bash "${SCRIPT_DIR}/lpm24/run_pipeline.sh" "${GPU_IDS}"
fi
echo ""

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo "=========================================="
echo "  Download Summary"
echo "=========================================="

check() {
    local name="$1" path="$2"
    if [ -f "${path}" ]; then
        local size
        size=$(du -h "${path}" | cut -f1)
        echo "  [OK] ${name} (${size})"
    elif [ -d "${path}" ]; then
        echo "  [OK] ${name} (directory)"
    else
        echo "  [--] ${name} (not found)"
    fi
}

check "ToyMix"       "data/graphium/neurips2023/small-dataset/qm9.csv"
check "LargeMix"     "data/graphium/neurips2023/large-dataset/PCQM4M_G25_N4.parquet"
check "RxRx3"        "data/rxrx3/rxrx3_smiles_embeddings.csv"
check "BBBC047"      "../../data/bbbc047/bbbc047_smiles_embeddings.csv"
check "DTI (ESM-2)"  "data/dti-processed/dti_esm2_100k.csv"
check "DTI (ESM-C)"  "data/dti-processed/dti_esmc_100k.csv"
check "LPM-24"       "data/dti-processed/lpm24_pubmedbert.csv"
echo ""
