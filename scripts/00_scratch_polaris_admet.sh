#!/usr/bin/env bash
# Train a model from scratch on the 6 Polaris biogen/adme-fang-v1 endpoints.
#
# Usage:
#   bash scripts/00_scratch_polaris_admet.sh <model> [gpu_id]
#
# Arguments:
#   model  : gcn | mpnn | gpspp | gpspp_800M | pairformer* | pairmixer* | pairmixerpp_12M
#   gpu_id : CUDA device index (default: 0)

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> [gpu_id]"}
DEVICE=${2:-${DEVICE}}

case "${MODEL}" in
    pairmixer_auto|pairmixer_12M|pairmixerpp_12M|pairmixer_10M|pairmixer_20M|pairmixer_40M|pairmixer_boltz)
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    gpspp_800M|pairformer|pairformer_17M|pairformer_52M|pairformer_boltz)
        BATCH_SIZE=${BATCH_SIZE:-32}
        ;;
    gcn|mpnn|gpspp)
        BATCH_SIZE=${BATCH_SIZE:-400}
        ;;
    *)
        echo "Error: unknown model '${MODEL}'."
        exit 1
        ;;
esac

TAGS="['${MODEL}','scratch','polaris_admet']"

echo "=== Training ${MODEL} from scratch on Polaris ADME-Fang (bs=${BATCH_SIZE}) ==="

for task in "${POLARIS_ADME_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        training=polaris_admet \
        tasks=polaris_admet \
        $(wandb_flags "${TAGS}") \
        ++constants.raise_train_error=False \
        ++constants.task=${task} \
        ++constants.name=polaris_${task}_${MODEL}_scratch \
        ++datamodule.args.polaris_benchmark_names=${task} \
        ++datamodule.args.batch_size_training=${BATCH_SIZE} \
        ++architecture.task_heads.${task}.hidden_dims=256 \
        ++architecture.task_heads.${task}.depth=4 \
        ++trainer.model_checkpoint.save_last=False \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
