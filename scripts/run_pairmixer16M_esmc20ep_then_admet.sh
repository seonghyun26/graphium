#!/usr/bin/env bash
# Pre-train PairMixer 16M on ToyMix + DTI (ESM-C v2) for 20 epochs, then
# fine-tune the newest produced checkpoint on all 22 ADMET tasks.
#
# Usage:
#   bash scripts/run_pairmixer16M_esmc20ep_then_admet.sh [gpu_id]
#
# Defaults:
#   gpu_id = 6

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_ID=${1:-6}
STAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR="$(pwd)/models_checkpoints/chain/pairmixer_16M_toymix_dti_esmc_v2_${STAMP}"

echo "=== [1/2] PairMixer 16M ESM-C v2 pre-train for 20 epochs on GPU ${GPU_ID} ==="
EXTRA_FLAGS="++constants.max_epochs=20 ++trainer.model_checkpoint.dirpath=${CKPT_DIR}/" \
    bash scripts/00_pretrain.sh pairmixer_16M toymix_dti_esmc_v2 "${GPU_ID}"

CKPT="$(find "${CKPT_DIR}" -name '*.ckpt' | sort | tail -1)"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "Error: no checkpoint produced under ${CKPT_DIR}" >&2
    exit 1
fi

echo "=== [2/2] PairMixer 16M ESM-C v2 -> ADMET fine-tune on GPU ${GPU_ID} ==="
echo "    ckpt: ${CKPT}"

PRETRAIN_DATASET=toymix_dti_esmc_v2 \
    bash scripts/00_finetune_admet.sh pairmixer_16M "${CKPT}" "${GPU_ID}"
