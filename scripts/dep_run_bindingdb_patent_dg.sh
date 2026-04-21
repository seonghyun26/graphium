#!/usr/bin/env bash
# End-to-end DTI-DG (BindingDB_Patent, temporal-split) eval for the TDC leaderboard.
#
# Stage 1: prep — downloads the dti_dg_group benchmark, joins ESM-C protein
#          embeddings, z-scores pY, writes data/dti-eval/BindingDB_Patent_DG.csv
#          and data/dti-eval/splits/BindingDB_Patent_DG_temporal_seed{0..4}.pt
# Stage 2: eval  — runs PairMixer + MiniMol + MolE through the sklearn head
#          (consistent with how each TDC-DTI-DG leaderboard entry reports PCC).
#
# Usage:
#   bash scripts/run_bindingdb_patent_dg.sh [gpu_id] [pairmixer_ckpt] [pairmixer_tag]
#
# Env vars (optional):
#   SEEDS="0 1 2 3 4"                         : space-sep seeds
#   HEAD=mlp                                  : linear | mlp
#   ENCODERS="pairmixer minimol mole"         : space-sep encoders to run
#   SKIP_PREP=1                               : skip stage 1 (re-use existing data)
#
# Results land in:
#   results/pairmixer_dti_results.csv  (rows with subset=BindingDB_Patent_DG, method=temporal)
#   results/minimol_dti_results.csv
#   results/mole_dti_results.csv
#
# Compare PCC (pearsonr column) against TDC's leaderboard:
#   https://tdcommons.ai/benchmark/dti_dg_group/bindingdb_patent/

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate graphium

GPU=${1:-7}
PAIRMIXER_CKPT=${2:-models_checkpoints/small-dataset/pairmixer_12M/2026-04-18_23-37-27_20260418_233727/toymix_pairmixer_12M_epochepoch=099_20260418_233727.ckpt}
PAIRMIXER_TAG=${3:-pairmixer12M_toymix}

SEEDS=${SEEDS:-"0 1 2 3 4"}
HEAD=${HEAD:-mlp}
ENCODERS=${ENCODERS:-"pairmixer minimol mole"}

# ── Stage 1: prep ────────────────────────────────────────────────────────────
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
    echo "============================================================"
    echo "[1/2] Prepping BindingDB_Patent_DG (temporal split, seeds: ${SEEDS})"
    echo "============================================================"
    python scripts/data/dti_eval/01_prepare_dti_eval.py \
        --subsets BindingDB_Patent_DG \
        --methods temporal \
        --seeds ${SEEDS}
fi

if [[ ! -f "data/dti-eval/BindingDB_Patent_DG.parquet" ]]; then
    echo "ERROR: data/dti-eval/BindingDB_Patent_DG.parquet not found after prep."
    exit 1
fi

# ── Stage 2: eval (sklearn head per encoder) ─────────────────────────────────
FAILED=()
run_eval() {
    local encoder=$1 seed=$2
    local script
    case "${encoder}" in
        pairmixer)
            echo ""
            echo "--- pairmixer / temporal / seed=${seed} ---"
            CUDA_VISIBLE_DEVICES=${GPU} python scripts/pairmixer/pairmixer_dti_eval.py \
                --ckpt "${PAIRMIXER_CKPT}" --ckpt-tag "${PAIRMIXER_TAG}" \
                --subset BindingDB_Patent_DG --method temporal --seed "${seed}" --head "${HEAD}" \
                || FAILED+=("pairmixer:seed${seed}")
            ;;
        minimol)
            echo ""
            echo "--- minimol / temporal / seed=${seed} ---"
            CUDA_VISIBLE_DEVICES=${GPU} python scripts/minimol/minimol_dti_eval.py \
                --subset BindingDB_Patent_DG --method temporal --seed "${seed}" --head "${HEAD}" \
                || FAILED+=("minimol:seed${seed}")
            ;;
        mole)
            echo ""
            echo "--- mole / temporal / seed=${seed} ---"
            CUDA_VISIBLE_DEVICES=${GPU} python scripts/mole/mole_dti_eval.py \
                --subset BindingDB_Patent_DG --method temporal --seed "${seed}" --head "${HEAD}" \
                || FAILED+=("mole:seed${seed}")
            ;;
        *)
            echo "ERROR: unknown encoder '${encoder}'"
            FAILED+=("unknown:${encoder}")
            ;;
    esac
}

echo ""
echo "============================================================"
echo "[2/2] Eval on GPU ${GPU} — encoders: ${ENCODERS}, head: ${HEAD}"
echo "============================================================"
for encoder in ${ENCODERS}; do
    for seed in ${SEEDS}; do
        run_eval "${encoder}" "${seed}"
    done
done

echo ""
echo "================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "=== ALL RUNS SUCCEEDED ==="
else
    echo "=== FAILED (${#FAILED[@]}): ${FAILED[*]} ==="
fi
echo "Results: results/{pairmixer,minimol,mole}_dti_results.csv"
echo "Compare PCC (pearsonr) vs TDC leaderboard:"
echo "  https://tdcommons.ai/benchmark/dti_dg_group/bindingdb_patent/"
