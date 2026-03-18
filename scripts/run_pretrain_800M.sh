#!/usr/bin/env bash
# Pre-train GPS++ 800M on all four datasets (one per GPU).
# Each run gets proper W&B tags for filtering.
#
# Usage:
#   bash scripts/run_pretrain_800M.sh
#
# Or run individually:
#   bash scripts/run_pretrain_800M.sh toymix 4
#   bash scripts/run_pretrain_800M.sh largemix 5
#   bash scripts/run_pretrain_800M.sh largemix_rxrx3 6
#   bash scripts/run_pretrain_800M.sh rxrx3 7

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${1:-all}
DEVICE=${2:-0}
SEED=${SEED:-0}

run_pretrain() {
    local dataset=$1
    local gpu=$2
    local tasks=$dataset
    local training=largemix
    local architecture=largemix

    if [[ "$dataset" == "toymix" ]]; then
        training=toymix
        architecture=toymix
    elif [[ "$dataset" == "rxrx3" ]]; then
        training=rxrx3
    fi

    echo "=== Pre-training GPS++ 800M on ${dataset} (GPU ${gpu}) ==="

    CUDA_VISIBLE_DEVICES=${gpu} graphium-train \
        model=gpspp_800M \
        accelerator=gpu \
        tasks=${tasks} \
        training=${training} \
        architecture=${architecture} \
        ++constants.seed=${SEED} \
        ++constants.wandb.entity=eddy26 \
        ++constants.wandb.save_dir=null \
        ++constants.wandb.project=graphium \
        "++constants.wandb.tags=['gpspp','gpspp_800M','pretrain','${dataset}']" \
        ++constants.wandb.name=pretrain_gpspp_800M_${dataset}
}

if [[ "$DATASET" == "all" ]]; then
    run_pretrain toymix 4 &
    run_pretrain largemix 5 &
    run_pretrain largemix_rxrx3 6 &
    run_pretrain rxrx3 7 &
    echo "All 4 pre-training jobs launched. Use 'nvidia-smi' to monitor."
    wait
else
    run_pretrain "$DATASET" "$DEVICE"
fi
