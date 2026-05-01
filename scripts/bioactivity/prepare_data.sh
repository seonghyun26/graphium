#!/usr/bin/env bash
# Build data/downstream/bioactivity/cell_bioactivity.csv from ChEMBL 33 + JUMP-CP.
#
# Usage:
#   bash scripts/bioactivity/prepare_data.sh
#   bash scripts/bioactivity/prepare_data.sh <path/to/chembl_33.db>
#
# Env overrides:
#   CHEMBL_DB=datacache/chembl/chembl_33/chembl_33_sqlite/chembl_33.db
#   JUMP_METADATA_DIR=/home/shpark/prj-molrepr/datacache/jump_cpcnn/metadata
#   OUT_DIR=data/downstream/bioactivity
#   SEED=0

set -euo pipefail
cd "$(dirname "$0")/../.."

eval "$(conda shell.bash hook)"
conda activate graphium-downstream 2>/dev/null || conda activate graphium

CHEMBL_DB=${1:-${CHEMBL_DB:-datacache/chembl/chembl_33/chembl_33_sqlite/chembl_33.db}}
JUMP_METADATA_DIR=${JUMP_METADATA_DIR:-/home/shpark/prj-molrepr/datacache/jump_cpcnn/metadata}
OUT_DIR=${OUT_DIR:-../data/downstream/bioactivity}
SEED=${SEED:-0}

if [[ ! -f "${CHEMBL_DB}" ]]; then
    echo "ERROR: ChEMBL DB not found at ${CHEMBL_DB}"
    echo "       Pass it explicitly: bash scripts/bioactivity/prepare_data.sh /path/to/chembl_33.db"
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "Preparing bioactivity data: ChEMBL ${CHEMBL_DB} + JUMP ${JUMP_METADATA_DIR}"
python scripts/data/bioactivity/01_prepare_bioactivity.py \
    --chembl-db "${CHEMBL_DB}" \
    --jump-metadata-dir "${JUMP_METADATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --seed "${SEED}"

echo
echo "Done. Ready for: bash scripts/bioactivity/run_all.sh [pairmixer_ckpt]"
