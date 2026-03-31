#!/usr/bin/env bash
# Train GPS++ (768d/8L) from scratch on 5 representative ADMET tasks — NO MoE.
# This is the same-sized model as the MoE pilot, providing an apples-to-apples
# comparison to isolate the effect of MoE on pre-training.
#
# Usage:
#   bash scripts/05_scratch_moe_baseline.sh [gpu_id]

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}

MODEL=gpspp
DIM=768
GNN_DEPTH=8
BATCH_SIZE=128

TAGS="['${MODEL}','scratch','admet','moe_baseline','768d8L']"

# Same 5 representative tasks as the MoE finetuning script
ABLATION_TASKS=(
    caco2_wang              # Absorption
    bbb_martins             # Distribution
    cyp3a4_veith            # Metabolism
    half_life_obach         # Excretion
    herg                    # Toxicity
)

echo "=== Scratch baseline: GPS++ 768d/8L (no MoE) on 5 ADMET tasks ==="
echo "  Model: ${MODEL} (dim=${DIM}, depth=${GNN_DEPTH})"
echo "  Tasks: ${ABLATION_TASKS[*]}"
echo "  GPU: ${DEVICE}"
echo ""

for task in "${ABLATION_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=admet \
        $(wandb_flags "${TAGS}") \
        ++constants.raise_train_error=False \
        ++constants.task=${task} \
        ++datamodule.args.tdc_benchmark_names=${task} \
        ++datamodule.args.num_workers=0 \
        ++datamodule.args.featurization_n_jobs=0 \
        ++architecture.gnn.depth=${GNN_DEPTH} \
        ++datamodule.args.batch_size_training=${BATCH_SIZE} \
        $(gpspp_dim_flags "${DIM}") \
        ++architecture.task_heads.${task}.hidden_dims=256 \
        ++architecture.task_heads.${task}.depth=4 \
        ++trainer.model_checkpoint.save_last=False \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done

echo ""
echo "Done. Results tagged with 'moe_baseline' in: results/experiment_results.csv"
