#!/usr/bin/env bash
# Re-run finetuning for excretion tasks only (with label normalization fix).
#
# Usage:
#   bash scripts/03_finetune_excretion.sh [gpu_id]
#
# Runs toymix, toymix_rxrx3, and toymix_dti checkpoints on:
#   half_life_obach, clearance_hepatocyte_az, clearance_microsome_az

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}
FINETUNE_DIM=${FINETUNE_DIM:-256}
ADDED_DEPTH=${ADDED_DEPTH:-4}

MODEL=gpspp_800M

EXCRETION_TASKS=(
    half_life_obach
    clearance_hepatocyte_az
    clearance_microsome_az
)

# Checkpoint paths (pretrain dataset → ckpt path, finetuning config)
declare -A CKPTS
declare -A FT_CONFIGS

CKPTS[toymix]="models_checkpoints/small-dataset/gpspp_800M/2026-03-18_14-53-57_20260318_145357/toymix_gpspp_800M_20260318_145357.ckpt"
FT_CONFIGS[toymix]="admet"

CKPTS[toymix_rxrx3]="models_checkpoints/toymix-rxrx3/gpspp_800M/2026-03-21_23-23-12_20260321_232312/toymix_rxrx3_gpspp_800M_20260321_232312.ckpt"
FT_CONFIGS[toymix_rxrx3]="admet_toymix_rxrx3"

CKPTS[toymix_dti]="models_checkpoints/toymix-dti/gpspp_800M/2026-03-21_22-57-10_20260321_225710/toymix_dti_gpspp_800M_20260321_225710.ckpt"
FT_CONFIGS[toymix_dti]="admet_toymix_dti"

UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}

for dataset in toymix toymix_rxrx3 toymix_dti; do
    CKPT="${CKPTS[$dataset]}"
    FINETUNING_CONFIG="${FT_CONFIGS[$dataset]}"
    TAGS="['${MODEL}','finetune','excretion_rerun','${dataset}']"

    echo ""
    echo "======================================================================"
    echo "  Dataset: ${dataset}"
    echo "  Checkpoint: ${CKPT}"
    echo "  Finetuning config: ${FINETUNING_CONFIG}"
    echo "======================================================================"

    for task in "${EXCRETION_TASKS[@]}"; do
        echo "--- Task: ${task} (${dataset}) ---"

        CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
            model=${MODEL} \
            accelerator=gpu \
            tasks=admet \
            $(wandb_flags "${TAGS}") \
            ++constants.raise_train_error=False \
            ++constants.task=${task} \
            ++finetuning.task=${task} \
            ++datamodule.args.tdc_benchmark_names=${task} \
            ++datamodule.args.num_workers=0 \
            ++datamodule.args.featurization_n_jobs=0 \
            +finetuning=${FINETUNING_CONFIG} \
            ++finetuning.pretrained_model=${CKPT} \
            ++finetuning.unfreeze_pretrained_depth=${UNFREEZE_DEPTH} \
            ++finetuning.epoch_unfreeze_all=${EPOCH_UNFREEZE_ALL} \
            ++finetuning.finetuning_head.in_dim=${FINETUNE_DIM} \
            ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
            ++finetuning.new_out_dim=${FINETUNE_DIM} \
            ++finetuning.finetuning_head.depth=${ADDED_DEPTH} \
            ++finetuning.added_depth=${ADDED_DEPTH} \
            ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
            ++trainer.model_checkpoint.save_last=False \
            ${EXTRA_FLAGS:-} \
        || echo "WARN: task ${task} (${dataset}) failed, continuing..."

        sleep 1
    done
done

echo ""
echo "Done. 3 datasets × 3 excretion tasks = 9 runs."
