#!/usr/bin/env bash
# Fine-tune the toymix-pretrained pairmixer_12M on all 22 ADMET benchmarks.
#
# Usage:
#   bash scripts/00_finetune_admet_pairmixer12M_toymix.sh [gpu_id] [ckpt]
#
# Defaults:
#   gpu_id : 0
#   ckpt   : latest toymix pairmixer_12M checkpoint under
#            models_checkpoints/small-dataset/pairmixer_12M/
#
# Env overrides (forwarded to 00_finetune_admet.sh):
#   FINETUNE_DIM, UNFREEZE_DEPTH, EPOCH_UNFREEZE_ALL, USE_COSINE, SEED, ...

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_ID=${1:-0}
CKPT=${2:-}

if [[ -z "${CKPT}" ]]; then
    CKPT=$(ls -t models_checkpoints/small-dataset/pairmixer_12M/*/toymix_pairmixer_12M_*.ckpt 2>/dev/null | head -n1 || true)
fi

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "Error: no toymix pairmixer_12M checkpoint found (looked under models_checkpoints/small-dataset/pairmixer_12M/)."
    echo "Pass one explicitly: $0 <gpu_id> <ckpt_path>"
    exit 1
fi

echo "=== pairmixer_12M (toymix pretrained) → ADMET on GPU ${GPU_ID} ==="
echo "    ckpt: ${CKPT}"

bash scripts/00_finetune_admet.sh pairmixer_12M "${CKPT}" "${GPU_ID}"
