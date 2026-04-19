#!/usr/bin/env bash
# PairMixer (toymix) pre-training stabilization sweep: pretrain -> 3-task ADMET finetune.
#
# Usage:
#   bash scripts/pairmixer_stabilize_sweep.sh <variant> <gpu>
#
# Variants: baseline | gradclip | accum | bundle
#     (each resolves a YAML under expts/hydra-configs/stabilize/)
#
# Outputs:
#   models_checkpoints/stabilize/<variant>/<timestamp>/last.ckpt
#   rows in results/experiment_results.csv tagged stabilize-<variant>
#   W&B run names: pairmixer_stab_<variant>_{pretrain,admet-<task>}

set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/common.sh"

VARIANT=${1:?  "Usage: $0 <variant> <gpu>"}
GPU=${2:?       "Usage: $0 <variant> <gpu>"}

MODEL=pairmixer_auto
DATASET=toymix
FT_TASKS="caco2_wang clearance_microsome_az ld50_zhu"
TS=$(date +%Y%m%d_%H%M%S)
CKPT_DIR="models_checkpoints/stabilize/${VARIANT}/${TS}"
RUN_NAME="pairmixer_stab_${VARIANT}"
LOG_DIR="logs/stabilize/${VARIANT}"
mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════"
echo "║ Stabilize sweep: variant=${VARIANT}  gpu=${GPU}  ts=${TS}"
echo "║ ckpt_dir=${CKPT_DIR}"
echo "╚══════════════════════════════════════════════════════════════"

# ── 1. Pretrain ───────────────────────────────────────────────────────────────
EXTRA_FLAGS="+stabilize=${VARIANT} \
    ++constants.name=${RUN_NAME} \
    ++trainer.model_checkpoint.dirpath=${CKPT_DIR}/ \
    ++constants.wandb.tags=['pairmixer_auto','pretrain','toymix','stabilize-${VARIANT}']" \
    DEVICE=${GPU} \
    bash scripts/00_pretrain.sh ${MODEL} ${DATASET} ${GPU} \
    2>&1 | tee "${LOG_DIR}/pretrain_${TS}.log"

# ── 2. Locate produced checkpoint ─────────────────────────────────────────────
# Hydra appends an extra timestamp suffix to dirpath; find the created leaf.
CKPT_PATH=$(find "${CKPT_DIR}" -name 'last.ckpt' -print -quit 2>/dev/null || true)
if [[ -z "${CKPT_PATH}" ]]; then
    # Fallback: glob for timestamp-suffixed sibling directories.
    CKPT_PATH=$(find "models_checkpoints/stabilize/${VARIANT}" -name 'last.ckpt' -newer "${LOG_DIR}/pretrain_${TS}.log" -print -quit)
fi
if [[ -z "${CKPT_PATH}" || ! -f "${CKPT_PATH}" ]]; then
    echo "ERROR: no last.ckpt found under models_checkpoints/stabilize/${VARIANT}" >&2
    exit 1
fi
echo ">>> Using checkpoint: ${CKPT_PATH}"

# ── 3. Fine-tune on 3 diagnostic ADMET tasks ──────────────────────────────────
ADMET_TASKS_OVERRIDE="${FT_TASKS}" \
    MODEL_TAG="stabilize-${VARIANT}" \
    DEVICE=${GPU} \
    bash scripts/00_finetune_admet.sh ${MODEL} "${CKPT_PATH}" ${GPU} \
    2>&1 | tee "${LOG_DIR}/finetune_${TS}.log"

echo ""
echo "=== Variant ${VARIANT} complete (${TS}) ==="
