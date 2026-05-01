#!/usr/bin/env bash
# MiniMol-style 2-stage probe on a frozen toymix+lpm24-LitOpenAI PairMixer 12M.
#
# Stage 1: 18-candidate HP sweep × 5 folds per task -> chosen-HP JSON.
# Stage 2: 5 reps × 5 folds at the swept HPs -> ensemble CSV row per task.
#
# Usage:
#   bash scripts/run_pairmixer12M_litopenai_probe.sh [gpu] [tasks...]
#
# Env vars (optional):
#   CKPT        : checkpoint path (default: 2026-04-24_22-23-52 last.ckpt)
#   REDO_SWEEP  : 1 to re-run Stage 1 even if a sweep JSON already exists
#                 (default: 1, since the previous JSON predates further training)
#
# Examples:
#   bash scripts/run_pairmixer12M_litopenai_probe.sh 1                # all 22 tasks
#   bash scripts/run_pairmixer12M_litopenai_probe.sh 1 caco2_wang     # single task
#   REDO_SWEEP=0 bash scripts/run_pairmixer12M_litopenai_probe.sh 1   # reuse stale sweep

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-1}
shift || true
TASKS=( "${@:-all}" )

CKPT=${CKPT:-models_checkpoints/toymix-lpm24-litopenai/pairmixer_12M/2026-04-24_22-23-52_20260424_222352/last.ckpt}
LABEL=toymix-lpm24-litopenai

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    echo "       Pass via CKPT=... or check the directory:"
    find models_checkpoints/toymix-lpm24-litopenai/pairmixer_12M -name '*.ckpt' 2>/dev/null | head -5
    exit 1
fi

export REDO_SWEEP=${REDO_SWEEP:-1}

echo "=== PairMixer 12M (toymix+lpm24+LitOpenAI) MiniMol probe ==="
echo "    GPU         : cuda:${GPU}"
echo "    ckpt        : ${CKPT}"
echo "    label       : ${LABEL}"
echo "    REDO_SWEEP  : ${REDO_SWEEP}"
echo "    tasks       : ${TASKS[*]}"
echo ""

bash scripts/01_pairmixer_minimol_probe.sh "${CKPT}" "${LABEL}" "${GPU}" "${TASKS[@]}"
