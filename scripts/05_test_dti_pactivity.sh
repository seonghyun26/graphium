#!/usr/bin/env bash
# DTI pActivity protein-label-based pre-training.
#
# New pre-training strategy:
#   MLP(GNN(molecule) || ESMC_protein_embedding) -> pActivity (-log10 of binding affinity)
#
# Usage:
#   bash scripts/05_test_dti_pactivity.sh [model] [dataset] [gpu_id]
#
# Arguments:
#   model   : gcn | pairmixer_20M | ... (default: pairmixer_20M)
#   dataset : dti_pactivity | toymix_dti_pactivity  (default: toymix_dti_pactivity)
#             dti_pactivity:        DTI-only (99k pairs)
#             toymix_dti_pactivity: ToyMix + DTI pActivity (qm9 + tox21 + zinc + DTI)
#   gpu_id  : CUDA device index (default: 5)
#
# Examples:
#   bash scripts/05_test_dti_pactivity.sh pairmixer_20M toymix_dti_pactivity 3
#   bash scripts/05_test_dti_pactivity.sh pairmixer_20M dti_pactivity 5
#   bash scripts/05_test_dti_pactivity.sh gcn toymix_dti_pactivity 5

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-pairmixer_20M}
DATASET=${2:-toymix_dti_pactivity}
DEVICE=${3:-5}

# ── Verify prerequisites ─────────────────────────────────────────────────────
DATA_CSV=data/dti-processed/dti_pactivity_esmc_100k.csv

if [[ ! -f "${DATA_CSV}" ]]; then
    echo "ERROR: Dataset not found: ${DATA_CSV}"
    echo "Run the prep script first:"
    echo "  python scripts/data/dti_esmc/06_prepare_dti_pactivity.py"
    exit 1
fi

# ── Activate conda env ───────────────────────────────────────────────────────
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "graphium" ]]; then
    # shellcheck disable=SC1091
    source /home/shpark/miniforge3/etc/profile.d/conda.sh
    conda activate graphium
fi

# ── Report ───────────────────────────────────────────────────────────────────
echo "=========================================================================="
echo "  DTI pActivity pre-training"
echo "=========================================================================="
echo "  Model:       ${MODEL}"
echo "  Dataset:     ${DATASET}"
echo "  GPU:         ${DEVICE}"
echo "  DTI task:    MLP(mol_repr || prot_emb_1152) -> pY"
echo "=========================================================================="
echo

bash scripts/00_pretrain.sh "${MODEL}" "${DATASET}" "${DEVICE}"
