#!/usr/bin/env bash
# Rerun pairformer_boltz scratch ADMET tasks that failed (empty metrics) on 2026-03-25.
# These tasks were skipped due to speed verification interruption.
#
# Usage:
#   bash scripts/02_scratch_admet_rerun.sh [gpu_id]

source "$(dirname "$0")/common.sh"

MODEL=pairformer_boltz
DEVICE=${1:-${DEVICE}}
DIM=${DIM:-384}
GNN_DEPTH=${GNN_DEPTH:-48}
BATCH_SIZE=${BATCH_SIZE:-32}

TAGS="['${MODEL}','scratch','admet']"

ARCH_FLAGS="++architecture.gnn.depth=${GNN_DEPTH}"
DATAMODULE_FLAGS="++datamodule.args.batch_size_training=${BATCH_SIZE}"

# Tasks that had empty metrics from the interrupted run
RERUN_TASKS=(
    cyp2c9_veith
    cyp2c9_substrate_carbonmangels
    cyp2d6_substrate_carbonmangels
    cyp3a4_substrate_carbonmangels
    half_life_obach
    clearance_hepatocyte_az
    clearance_microsome_az
    ld50_zhu
    herg
    ames
    dili
)

echo "=== Rerunning ${MODEL} scratch on ${#RERUN_TASKS[@]} failed ADMET tasks ==="

for task in "${RERUN_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=admet \
        $(wandb_flags "${TAGS}") \
        ++constants.raise_train_error=False \
        ++constants.task=${task} \
        ++datamodule.args.tdc_benchmark_names=${task} \
        ${ARCH_FLAGS} \
        ${DATAMODULE_FLAGS} \
        ++architecture.task_heads.${task}.hidden_dims=256 \
        ++architecture.task_heads.${task}.depth=4 \
        ++trainer.model_checkpoint.save_last=False \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
