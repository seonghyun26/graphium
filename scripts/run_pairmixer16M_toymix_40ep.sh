#!/usr/bin/env bash
# Pre-train PairMixer 16M on ToyMix for 40 epochs.
#
# Usage:
#   bash scripts/run_pairmixer16M_toymix_40ep.sh [gpu_id]
#
# Defaults:
#   gpu_id = 6

source "$(dirname "$0")/common.sh"

DEVICE=${1:-6}

EXTRA_FLAGS="++constants.max_epochs=40 ${EXTRA_FLAGS:-}" \
    bash "$(dirname "$0")/00_pretrain.sh" pairmixer_16M toymix "${DEVICE}"
