#!/usr/bin/env bash
# Linear-probe a toymix-pretrained PairMixer 12M on the TDC DTI eval setup.
#
# Loads a toymix-only PairMixer 12M checkpoint, freezes the molecule encoder,
# and trains a fresh `MLP([z_mol || prot_emb_1152]) -> y` head from scratch on
# each TDC DTI subset / split / seed.
#
# Default scope: DAVIS + KIBA × random + cold_target × seeds 0..4 = 20 runs on GPU 6.
#
# Usage:
#   bash scripts/run_pairmixer12M_dti_toymix.sh [gpu_id] [checkpoint_path]
#
# Env vars (optional):
#   CKPT       : pretrained checkpoint path (default: latest small-dataset/pairmixer_12M ckpt)
#   SUBSETS    : comma-sep subsets         (default: DAVIS,KIBA)
#   METHODS    : comma-sep split methods   (default: random,cold_target)
#   SEEDS      : comma-sep seeds           (default: 0,1,2,3,4)
#   DTI_EVAL_DIR : data prep root          (default: data/dti-eval)
#
# Wraps scripts/00_finetune_dti.sh with FINETUNING_CONFIG=dti_eval_from_toymix
# (which uses sub_module_from_pretrained=zinc + module_overrides to give the
# rebuilt head the 1152-d protein aux input that toymix heads don't have).

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE=${1:-6}
CKPT=${2:-${CKPT:-models_checkpoints/small-dataset/pairmixer_12M/2026-04-18_23-37-27_20260418_233727/toymix_pairmixer_12M_epochepoch=099_20260418_233727.ckpt}}

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    echo "       Pass a path as the 2nd arg or via CKPT=..."
    echo "       Available toymix-only PairMixer 12M ckpts:"
    find models_checkpoints/small-dataset/pairmixer_12M -name '*.ckpt' 2>/dev/null | head -5
    exit 1
fi

export FINETUNING_CONFIG=dti_eval_from_toymix
export DTI_SUBSETS=${SUBSETS:-DAVIS,KIBA}
export DTI_METHODS=${METHODS:-random,cold_target}
export DTI_SEEDS=${SEEDS:-0,1,2,3,4}
export DTI_EVAL_DIR=${DTI_EVAL_DIR:-data/dti-eval}

echo "=== PairMixer 12M (toymix-pretrained, frozen encoder) on DTI eval ==="
echo "    GPU     : ${DEVICE}"
echo "    ckpt    : ${CKPT}"

bash scripts/00_finetune_dti.sh pairmixer_12M "${CKPT}" "${DEVICE}"
