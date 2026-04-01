#!/usr/bin/env bash
# Rerun pairformer_boltz finetune on ADMET tasks that were interrupted during
# the rxrx3 pre-trained model evaluation (stopped after cyp3a4_veith).
#
# Usage:
#   bash scripts/01_finetune_admet_rxrx3_pairformer_rerun.sh [gpu_id]

source "$(dirname "$0")/common.sh"

MODEL=pairformer_boltz
DEVICE=${1:-${DEVICE}}
CKPT="./models_checkpoints/rxrx3/pairformer_boltz/2026-03-23_16-49-24_20260323_164925/last.ckpt"

# Override the task list to only the 11 that were missed
ADMET_TASKS=(
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

echo "=== Rerunning pairformer_boltz rxrx3→ADMET finetune on ${#ADMET_TASKS[@]} tasks (GPU ${DEVICE}) ==="

# Detect pretrain dataset and finetuning config
PRETRAIN_DATASET="rxrx3"
FINETUNING_CONFIG="admet_rxrx3"

UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}
FINETUNE_DIM=${FINETUNE_DIM:-256}
ADDED_DEPTH=${ADDED_DEPTH:-4}
TAGS="['${MODEL}','finetune','admet','${PRETRAIN_DATASET}']"

DIM=${DIM:-384}
GNN_DEPTH=${GNN_DEPTH:-48}
BATCH_SIZE=${BATCH_SIZE:-32}
ARCH_FLAGS="++architecture.gnn.depth=${GNN_DEPTH}"
DATAMODULE_FLAGS="++datamodule.args.batch_size_training=${BATCH_SIZE}"

for task in "${ADMET_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=admet \
        +finetuning=${FINETUNING_CONFIG} \
        $(wandb_flags "${TAGS}") \
        ++constants.task=${task} \
        ++finetuning.task=${task} \
        ++finetuning.pretrained_model="${CKPT}" \
        ++finetuning.unfreeze_pretrained_depth=${UNFREEZE_DEPTH} \
        ++finetuning.epoch_unfreeze_all=${EPOCH_UNFREEZE_ALL} \
        ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
        ++finetuning.added_depth=${ADDED_DEPTH} \
        ++datamodule.args.tdc_benchmark_names=${task} \
        ${ARCH_FLAGS} \
        ${DATAMODULE_FLAGS} \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
