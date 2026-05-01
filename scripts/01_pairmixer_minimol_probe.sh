#!/usr/bin/env bash
# Two-stage MiniMol-style probe on a frozen PairMixer 12M backbone.
#
# Stage 1 — HP sweep (18 candidates × K folds × 1 rep, MiniMol "Algorithm 1"):
#   For each task, picks the lr/hidden_dim/depth combo with the smallest
#   mean val loss across K folds. Writes the chosen HPs to a JSON file.
#
# Stage 2 — final ensemble (5 reps × 5 folds = 25 models per task at the
# chosen HP): logit-mean across the 5 fold-models within each rep, mean ±
# std across the 5 reps. Writes one row per task to the ensemble CSV.
#
# Usage:
#   bash scripts/01_pairmixer_minimol_probe.sh <ckpt> <pretrain_label> [gpu] [tasks]
#
# Arguments:
#   ckpt            : .ckpt path for the frozen PairMixer 12M backbone
#   pretrain_label  : free-form label for this ckpt's pretraining recipe
#                     (becomes the ``pretrain`` column in the CSV; the
#                     sweep JSON is keyed off it too).
#   gpu             : CUDA device index (default: 0)
#   tasks           : space-separated TDC ADMET task list   (default: caco2_wang)
#
# Examples:
#   # caco2_wang only, GPU 5
#   bash scripts/01_pairmixer_minimol_probe.sh \
#        models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/2026-04-18_11-23-58_20260418_112358/last.ckpt \
#        toymix-dti-esmc-v2 5
#
#   # two arms in parallel on GPUs 5 and 7:
#   bash scripts/01_pairmixer_minimol_probe.sh \
#        models_checkpoints/small-dataset/pairmixer_12M/.../last.ckpt small-dataset 5 caco2_wang &
#   bash scripts/01_pairmixer_minimol_probe.sh \
#        models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/.../last.ckpt toymix-dti-esmc-v2 7 caco2_wang &
#   wait
#
#   # all 22 ADMET tasks (drop --tasks to default to the full set):
#   bash scripts/01_pairmixer_minimol_probe.sh <ckpt> <label> <gpu> all
#
# Outputs:
#   results/pairmixer_minimol_probe_sweep_<label>.json   per-task best HP
#   results/pairmixer_minimol_probe_sweep.csv            per-candidate val loss
#   results/pairmixer_minimol_probe_ensemble.csv         final mean ± std per task
#   logs/pairmixer_probe_<label>_<stamp>/{sweep,eval}.log  one log file per stage
#
# The script skips Stage 1 if a sibling sweep JSON already exists (set
# ``REDO_SWEEP=1`` to re-run it). It always runs Stage 2.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
PYTHON=/home/shpark/.conda/envs/graphium/bin/python
PROBE=scripts/pairmixer/pairmixer_minimol_probe.py

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ckpt> <pretrain_label> [gpu] [tasks]" >&2
    exit 1
fi
CKPT=$1
LABEL=$2
GPU=${3:-0}
shift 2 || true
shift || true   # consume optional gpu
TASKS_RAW=${*:-caco2_wang}

if [[ "${TASKS_RAW}" == "all" ]]; then
    TASKS_FLAGS=()   # no --tasks → script defaults to all 22
else
    # shellcheck disable=SC2206
    TASKS_ARR=( ${TASKS_RAW} )
    TASKS_FLAGS=(--tasks "${TASKS_ARR[@]}")
fi

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: ckpt not found: ${CKPT}" >&2
    exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_DIR}/logs/pairmixer_probe_${LABEL}_${STAMP}"
mkdir -p "${LOG_DIR}"

SWEEP_JSON="${REPO_DIR}/results/pairmixer_minimol_probe_sweep_${LABEL}.json"

cat <<EOF
== MiniMol-style PairMixer 12M probe ==
  pretrain label : ${LABEL}
  ckpt           : ${CKPT}
  GPU            : cuda:${GPU}
  tasks          : ${TASKS_RAW}
  log dir        : ${LOG_DIR}/
  sweep JSON     : ${SWEEP_JSON}
  ensemble CSV   : results/pairmixer_minimol_probe_ensemble.csv

Launching in 5 s — Ctrl-C to abort.
EOF
sleep 5

# ── Stage 1: HP sweep ────────────────────────────────────────────────────────
if [[ -f "${SWEEP_JSON}" && "${REDO_SWEEP:-0}" != "1" ]]; then
    echo ">> Stage 1 SKIPPED (sweep JSON already exists; set REDO_SWEEP=1 to overwrite)"
else
    echo ""
    echo ">> Stage 1: HP sweep (18 candidates per task) -> ${LOG_DIR}/sweep.log"
    "${PYTHON}" "${PROBE}" \
        --ckpt "${CKPT}" \
        --pretrain-label "${LABEL}" \
        --device "cuda:${GPU}" \
        --sweep \
        "${TASKS_FLAGS[@]}" \
        2>&1 | tee "${LOG_DIR}/sweep.log"
    rc1=${PIPESTATUS[0]}
    if [[ ${rc1} -ne 0 ]]; then
        echo "!! Stage 1 failed (exit ${rc1}); not running Stage 2." >&2
        exit ${rc1}
    fi
fi

# ── Stage 2: 5×5 ensemble at the swept HPs ───────────────────────────────────
echo ""
echo ">> Stage 2: 5x5 ensemble at swept HPs -> ${LOG_DIR}/eval.log"
"${PYTHON}" "${PROBE}" \
    --ckpt "${CKPT}" \
    --pretrain-label "${LABEL}" \
    --device "cuda:${GPU}" \
    "${TASKS_FLAGS[@]}" \
    2>&1 | tee "${LOG_DIR}/eval.log"
rc2=${PIPESTATUS[0]}
if [[ ${rc2} -ne 0 ]]; then
    echo "!! Stage 2 failed (exit ${rc2})." >&2
    exit ${rc2}
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "== Done =="
echo "  sweep JSON   : ${SWEEP_JSON}"
echo "  ensemble CSV : results/pairmixer_minimol_probe_ensemble.csv"
echo "  notebook     : notebooks/03_test_ensemble_results.ipynb"
exit 0
