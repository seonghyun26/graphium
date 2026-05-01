#!/usr/bin/env bash
# ADMET N-seed ensemble sweep — MiniMol-style probability ensembling.
#
# Iterates SEEDS × all 22 ADMET tasks for one (PRETRAIN, MODEL) combination.
# Each per-seed graphium-train run dumps its raw test predictions to
# ``outputs/<timestamp>_ft_<task>/predictions.pt`` via the predictor patch
# (env var GRAPHIUM_DUMP_PREDS=1). Aggregate post-hoc with
# ``scripts/aggregate_admet_ensemble.py``.
#
# Usage
# -----
#   GPU=6 PRETRAIN=scratch bash scripts/00_finetune_admet_ensemble.sh
#
#   GPU=5 PRETRAIN=toymix \
#       CKPT=models_checkpoints/small-dataset/pairmixer_12M/.../last.ckpt \
#       bash scripts/00_finetune_admet_ensemble.sh
#
#   GPU=4 PRETRAIN=toymix_dti_esmc_v2 \
#       CKPT=models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/.../last.ckpt \
#       bash scripts/00_finetune_admet_ensemble.sh
#
# Env overrides
# -------------
#   GPU        GPU index                           (default: 0)
#   PRETRAIN   'scratch' | label describing the ckpt source (REQUIRED)
#              Also becomes the ``method`` wandb tag + results-CSV column.
#   CKPT       .ckpt path (REQUIRED unless PRETRAIN=scratch)
#   MODEL      architecture keyword                (default: pairmixer_12M)
#   SEEDS      space-sep seed list                 (default: "0 1 2 3 4 5 6 7 8 9")
#   TASKS      space-sep subset of ADMET_TASKS     (default: all 22 via common.sh)
#   EXTRA_FLAGS  appended to each graphium-train call
#
# Notes
# -----
# - Scale: SEEDS × 22 tasks ≈ 220 runs per PRETRAIN. Budget accordingly.
# - Run in tmux / screen. Partition across GPUs by launching this script once
#   per PRETRAIN on a different GPU.
# - Invokes the existing ``00_{scratch,finetune}_admet.sh`` per seed — no
#   wrapping, just a seed loop that sets SEED + GRAPHIUM_DUMP_PREDS env vars.

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
MODEL=${MODEL:-pairmixer_12M}
PRETRAIN=${PRETRAIN:?"ERROR: must set PRETRAIN=scratch|toymix|toymix_dti_esmc_v2|..."}
CKPT=${CKPT:-}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
TASKS=${TASKS:-}  # empty = use common.sh default (all 22)
EXTRA_FLAGS=${EXTRA_FLAGS:-}

if [[ "${PRETRAIN}" == "scratch" ]]; then
    BASE_SCRIPT=scripts/00_scratch_admet.sh
    if [[ -n "${CKPT}" ]]; then
        echo "WARN: PRETRAIN=scratch but CKPT is set; the base script ignores it."
    fi
    BASE_ARGS=("${MODEL}" "${GPU}")
else
    BASE_SCRIPT=scripts/00_finetune_admet.sh
    if [[ -z "${CKPT}" ]]; then
        echo "ERROR: PRETRAIN=${PRETRAIN} requires CKPT=<path>." >&2
        exit 1
    fi
    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: CKPT not found: ${CKPT}" >&2
        exit 1
    fi
    BASE_ARGS=("${MODEL}" "${CKPT}" "${GPU}")
fi

# Propagate a task subset by exposing ADMET_TASKS_OVERRIDE (common.sh reads it).
TASKS_ENV=()
if [[ -n "${TASKS}" ]]; then
    TASKS_ENV=(ADMET_TASKS_OVERRIDE="${TASKS}")
fi

# Per-pretrain results/cache discrimination happens automatically via the base
# scripts' PRETRAIN_DATASET auto-detect from CKPT path (finetune) or the
# 'Scratch' method tag (scratch). No extra flags needed.

echo "============================================================"
echo "  ADMET ensemble sweep"
echo "  PRETRAIN=${PRETRAIN}  MODEL=${MODEL}  GPU=${GPU}"
echo "  SEEDS=[${SEEDS}]"
echo "  TASKS=[${TASKS:-<default:22>}]"
echo "============================================================"

for seed in ${SEEDS}; do
    echo
    echo "────── seed=${seed} ──────"
    SEED="${seed}" \
    GRAPHIUM_DUMP_PREDS=1 \
    EXTRA_FLAGS="${EXTRA_FLAGS}" \
    "${TASKS_ENV[@]}" \
        bash "${BASE_SCRIPT}" "${BASE_ARGS[@]}" \
        || echo "WARN: seed=${seed} had failures; continuing next seed"
done

echo
echo "Done. Per-seed predictions dumped under outputs/<ts>_ft_<task>/predictions.pt"
echo "Aggregate with:  python scripts/aggregate_admet_ensemble.py --pretrain ${PRETRAIN}"
