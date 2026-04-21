#!/usr/bin/env bash
# Regularization sweep for caco2_wang on the toymix-dti-esmc-v2 pairmixer_12M
# checkpoint with multi-layer pair pooling ([3, 7]) and a frozen backbone.
#
# Baseline is the current `admet_toymix_dti_pairpool` defaults (dropout=0.18
# inherited from pretrain, lr=1e-4, wd=0, 100 epochs). Each variant changes
# one or more of {dropout, lr, weight_decay, max_epochs}.
#
# Seeds: 0, 1, 2 per config (small ADMET is noisy — single-seed runs mislead).
#
# Usage:
#   bash scripts/caco2_pairpool_regsweep.sh [gpu_id]   # default: 5

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate graphium 2>/dev/null || true
export PATH="/home/shpark/.conda/envs/graphium/bin:$PATH"

DEVICE=${1:-5}

MODEL=pairmixer_12M
CKPT="./models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/2026-04-21_14-03-55_20260421_140355/last.ckpt"
TASK=caco2_wang

LOG_DIR="logs/caco2_pairpool_regsweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "Logs → ${LOG_DIR}"
echo "GPU=${DEVICE}, model=${MODEL}"

run_variant () {
    local name="$1"
    local seed="$2"
    local extra="$3"

    local run_name="${name}_seed${seed}"
    echo "=========================================="
    echo "  [${run_name}]  $(date)"
    echo "  extra: ${extra}"
    echo "=========================================="

    ADMET_TASKS_OVERRIDE=${TASK} \
    UNFREEZE_DEPTH=0 \
    EPOCH_UNFREEZE_ALL=none \
    FINETUNING_CONFIG=admet_toymix_dti_pairpool \
    MODEL_TAG="pair_regsweep_${name}" \
    EXTRA_FLAGS="++constants.seed=${seed} ${extra}" \
    bash scripts/00_finetune_admet.sh ${MODEL} "${CKPT}" ${DEVICE} \
        2>&1 | tee "${LOG_DIR}/${run_name}.log" \
        || echo "WARN: ${run_name} failed, continuing..."

    echo ">>> ${run_name} done at $(date)" >> "${LOG_DIR}/summary.txt"
}

# ── Variants ──────────────────────────────────────────────────────────────────
# Baseline: no extra regularization (uses YAML defaults).
BASELINE=""

# Dropout-only levers.
DROPOUT_04='++finetuning.module_overrides.dropout=0.4 ++finetuning.keep_modules_after_finetuning_module.task_heads-dti.dropout=0.4 ++finetuning.finetuning_head.dropout=0.4'
DROPOUT_06='++finetuning.module_overrides.dropout=0.6 ++finetuning.keep_modules_after_finetuning_module.task_heads-dti.dropout=0.6 ++finetuning.finetuning_head.dropout=0.6'

# Dropout 0.4 + weight decay.
DROPOUT_04_WD="${DROPOUT_04} ++predictor.optim_kwargs.weight_decay=0.01"

# Dropout 0.4 + weight decay + lower LR.
DROPOUT_04_WD_LOWLR="${DROPOUT_04_WD} ++predictor.optim_kwargs.lr=5e-5"

# Dropout 0.4 + weight decay + shorter schedule (cosine T_max follows max_epochs).
DROPOUT_04_WD_SHORT="${DROPOUT_04_WD} ++constants.max_epochs=40"

# All three: dropout 0.4 + wd + low lr + short schedule.
DROPOUT_04_ALL="${DROPOUT_04_WD_LOWLR} ++constants.max_epochs=40"

# ── Execute sweep ─────────────────────────────────────────────────────────────
echo "Started caco2_wang pair-pool regsweep at $(date)" | tee "${LOG_DIR}/summary.txt"

for seed in 0 1 2; do
    run_variant baseline              ${seed} "${BASELINE}"
    run_variant dropout04             ${seed} "${DROPOUT_04}"
    run_variant dropout06             ${seed} "${DROPOUT_06}"
    run_variant dropout04_wd          ${seed} "${DROPOUT_04_WD}"
    run_variant dropout04_wd_lowlr    ${seed} "${DROPOUT_04_WD_LOWLR}"
    run_variant dropout04_wd_short    ${seed} "${DROPOUT_04_WD_SHORT}"
    run_variant dropout04_all         ${seed} "${DROPOUT_04_ALL}"
done

echo ""
echo "============================================="
echo "  SWEEP COMPLETE at $(date)"
echo "  Logs: ${LOG_DIR}"
echo "============================================="
