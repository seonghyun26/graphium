#!/usr/bin/env bash
# Build data/downstream/gram_dti/*.parquet + splits from a DTIAM clone + ESM-2.
#
# Stages:
#   1. Collect (Target_ID, Target) for proteins actually used by the benchmark.
#   2. Run the existing ESM-2 extractor against that CSV (produces per-protein .pt).
#   3. Consolidate to a single parquet keyed by protein_id.
#   4. Sample 1:10 negatives, z-score protein features, emit parquet + fold splits.
#
# Usage:
#   bash scripts/gram_dti/prepare_data.sh <path/to/DTIAM_clone> [gpu_ids]
#
# Example:
#   git clone --depth 1 https://github.com/CSUBioGroup/DTIAM /tmp/DTIAM
#   bash scripts/gram_dti/prepare_data.sh /tmp/DTIAM 0,1

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate graphium-downstream 2>/dev/null || conda activate graphium

DTIAM_ROOT=${1:?"Usage: $0 <DTIAM_clone> [gpu_ids]"}
GPUS=${2:-0}

OUT_DIR="../data/downstream/gram_dti"
PROTEIN_CSV="${OUT_DIR}/dtiam-proteins.csv"
ESM2_OUT="${OUT_DIR}/protein-esm2"
PROTEIN_PARQUET="${OUT_DIR}/protein-esm2.parquet"

mkdir -p "${OUT_DIR}"

# ── 1. Collect proteins ──────────────────────────────────────────────────────
echo "[1/4] Collecting DTIAM proteins -> ${PROTEIN_CSV}"
python scripts/data/gram_dti/01_collect_proteins.py \
    --dtiam-root "${DTIAM_ROOT}" \
    --output "${PROTEIN_CSV}"

# ── 2. Extract ESM-2 (reuses the existing dti_esm2 extractor) ────────────────
# GRAM-DTI paper: esm2_t33_650M_UR50D, layer 33, 1280-d.
echo "[2/4] Extracting ESM-2 (650M / layer 33 / 1280-d) on GPUs ${GPUS}"
python scripts/data/dti_esm2/02_extract_embeddings.py \
    --input "${PROTEIN_CSV}" \
    --output-dir "${ESM2_OUT}" \
    --gpus "${GPUS}" \
    --esm-model esm2_t33_650M_UR50D \
    --repr-layer 33

# ── 3. Consolidate per-protein .pt -> single parquet ─────────────────────────
echo "[3/4] Consolidating -> ${PROTEIN_PARQUET}"
python scripts/data/gram_dti/02_consolidate_esm2.py \
    --embedding-dir "${ESM2_OUT}/embedding" \
    --protein-csv "${PROTEIN_CSV}" \
    --output "${PROTEIN_PARQUET}" \
    --repr-layer 33

# ── 4. Build per-subset parquets + per-fold splits ───────────────────────────
echo "[4/4] Building subset parquets + splits"
python scripts/data/gram_dti/03_prepare_parquet.py \
    --dtiam-root "${DTIAM_ROOT}" \
    --protein-emb "${PROTEIN_PARQUET}" \
    --output-dir "${OUT_DIR}"

echo
echo "Done. Ready for: bash scripts/gram_dti/run_all.sh [pairmixer_ckpt]"
