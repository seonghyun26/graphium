#!/usr/bin/env bash
# Build data/downstream/tdc_dti_regression/*.parquet + per-seed splits.
#
# Relies on the existing ESM-2 extractor at scripts/data/dti_esm2/ to produce
# per-protein .pt files, then consolidates + joins against TDC's DTI loaders.
#
# Usage:
#   bash scripts/tdc_dti_regression/prepare_data.sh [gpu_ids]
#
# Env overrides:
#   SUBSETS="DAVIS KIBA"                    # space-separated
#   METHODS="random cold_target"
#   SEEDS="0 1 2 3 4"
#   DG_SUBSETS=""                           # set to "BindingDB_Patent_DG" for temporal

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate graphium-downstream 2>/dev/null || conda activate graphium

GPUS=${1:-0}
SUBSETS=${SUBSETS:-"DAVIS KIBA"}
METHODS=${METHODS:-"random cold_target"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DG_SUBSETS=${DG_SUBSETS:-""}

OUT_DIR="../data/downstream/tdc_dti_regression"
PROTEIN_CSV="${OUT_DIR}/protein-drug.csv"
ESM2_OUT="${OUT_DIR}/protein-esm2"
PROTEIN_PARQUET="${OUT_DIR}/protein-esm2.parquet"

mkdir -p "${OUT_DIR}"

# ── 1. Collect TDC DTI proteins ──────────────────────────────────────────────
# Delegate to the existing collector (writes Target_ID, Target, Drug, Y, ...).
if [[ ! -f "${PROTEIN_CSV}" ]]; then
    echo "[1/4] Collecting TDC DTI data -> ${PROTEIN_CSV}"
    python scripts/data/dti_esm2/01_collect_tdc_data.py \
        --output "${PROTEIN_CSV}" --cache-dir "data/tdc_cache"
else
    echo "[1/4] ${PROTEIN_CSV} already exists; skipping collection."
fi

# ── 2. Extract ESM-2 (reuses the existing extractor) ─────────────────────────
# Default: esm2_t33_650M_UR50D / layer 33 / 1280-d (matches GRAM-DTI task).
echo "[2/4] Extracting ESM-2 (650M / layer 33 / 1280-d) on GPUs ${GPUS}"
python scripts/data/dti_esm2/02_extract_embeddings.py \
    --input "${PROTEIN_CSV}" \
    --output-dir "${ESM2_OUT}" \
    --gpus "${GPUS}" \
    --esm-model esm2_t33_650M_UR50D \
    --repr-layer 33

# ── 3. Consolidate into a single parquet ─────────────────────────────────────
echo "[3/4] Consolidating -> ${PROTEIN_PARQUET}"
python scripts/data/gram_dti/02_consolidate_esm2.py \
    --embedding-dir "${ESM2_OUT}/embedding" \
    --protein-csv "${PROTEIN_CSV}" \
    --output "${PROTEIN_PARQUET}" \
    --repr-layer 33

# ── 4. Per-subset parquet + per-seed splits ──────────────────────────────────
echo "[4/4] Building regular subsets: ${SUBSETS}"
python scripts/data/tdc_dti_regression/01_prepare_parquet.py \
    --protein-emb "${PROTEIN_PARQUET}" \
    --output-dir "${OUT_DIR}" \
    --subsets ${SUBSETS} \
    --methods ${METHODS} \
    --seeds ${SEEDS}

if [[ -n "${DG_SUBSETS}" ]]; then
    echo "[4b] Building DG subsets (temporal split): ${DG_SUBSETS}"
    python scripts/data/tdc_dti_regression/01_prepare_parquet.py \
        --protein-emb "${PROTEIN_PARQUET}" \
        --output-dir "${OUT_DIR}" \
        --subsets ${DG_SUBSETS} \
        --methods temporal \
        --seeds ${SEEDS}
fi

echo
echo "Done. Ready for: bash scripts/tdc_dti_regression/run_all.sh [pairmixer_ckpt]"
