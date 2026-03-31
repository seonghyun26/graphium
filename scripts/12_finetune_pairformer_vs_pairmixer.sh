#!/usr/bin/env bash
# Fine-tune pairformer_boltz and pairmixer_boltz (both toymix-pretrained) on
# 5 representative ADMET tasks to compare downstream performance.
#
# Usage:
#   bash scripts/12_finetune_pairformer_vs_pairmixer.sh [gpu_id]

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}
FINETUNE_DIM=${FINETUNE_DIM:-256}
ADDED_DEPTH=${ADDED_DEPTH:-4}
UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}
FINETUNING_CONFIG="admet"  # sub_module: zinc (toymix pretrained)

# 5 representative tasks (one per ADMET category)
ABLATION_TASKS=(
    caco2_wang              # Absorption
    bbb_martins             # Distribution
    cyp3a4_veith            # Metabolism
    half_life_obach         # Excretion
    herg                    # Toxicity
)

# Latest toymix-pretrained checkpoints
declare -A MODELS
MODELS[pairformer_boltz]="models_checkpoints/small-dataset/pairformer_boltz/2026-03-23_14-53-45_20260323_145345/toymix_pairformer_boltz_20260323_145345.ckpt"
MODELS[pairmixer_boltz]="models_checkpoints/small-dataset/pairmixer_boltz/2026-03-31_10-55-09_20260331_105509/toymix_pairmixer_boltz_20260331_105509.ckpt"

echo "=== Pairformer vs Pairmixer: downstream ADMET finetuning ==="
echo "  Tasks: ${ABLATION_TASKS[*]}"
echo "  GPU: ${DEVICE}"
echo ""

for MODEL in pairformer_boltz pairmixer_boltz; do
    CKPT="${MODELS[$MODEL]}"

    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: Checkpoint not found for ${MODEL}: ${CKPT}"
        continue
    fi

    TAGS="['${MODEL}','finetune','admet','toymix','arch_comparison']"

    echo "============================================================"
    echo "  ${MODEL}"
    echo "  Checkpoint: ${CKPT}"
    echo "============================================================"

    for task in "${ABLATION_TASKS[@]}"; do
        echo "--- Task: ${task} (${MODEL}) ---"

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
        || echo "WARN: task ${task} (${MODEL}) failed, continuing..."

        sleep 1
    done

    echo ""
done

echo "=== Done: 2 models × 5 tasks = 10 runs ==="
echo "Results tagged with 'arch_comparison' in: results/experiment_results.csv"
