#!/usr/bin/env bash
# Fine-tune a PairMixer 16M checkpoint on all 22 ADMET tasks.
#
# Usage:
#   bash scripts/run_pairmixer16M_finetune_admet.sh <checkpoint> [gpu_id]
#
# Defaults:
#   gpu_id = 6

source "$(dirname "$0")/common.sh"

CKPT=${1:? "Usage: $0 <checkpoint> [gpu_id]"}
DEVICE=${2:-6}

bash "$(dirname "$0")/00_finetune_admet.sh" pairmixer_16M "${CKPT}" "${DEVICE}"
