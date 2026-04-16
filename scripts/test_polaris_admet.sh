#!/usr/bin/env bash
# Quick smoke test: Polaris ADMET pipeline with pairmixer_10M (single task, 5 epochs).
# Verifies: data download, training loop, evaluation metrics, results CSV.
#
# Usage:
#   bash scripts/test_polaris_admet.sh [gpu_id]

set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure conda env is active
eval "$(conda shell.bash hook)"
conda activate graphium

DEVICE=${1:-7}
MODEL=pairmixer_20M
CKPT=models_checkpoints/small-dataset/pairmixer_20M/2026-04-13_19-42-52_20260413_194252/last.ckpt
SEED=0
FINETUNE_DIM=256
ADDED_DEPTH=4

POLARIS_TASKS=(
    adme_fang_hclint
    adme_fang_rclint
    adme_fang_perm
    adme_fang_hppb
    adme_fang_rppb
    adme_fang_solu
)

echo "=== Polaris ADMET smoke test (all 6 tasks) ==="
echo "Model: ${MODEL}, GPU: ${DEVICE}, Epochs: 5"

FAILED_TASKS=()

for TASK in "${POLARIS_TASKS[@]}"; do
    echo ""
    echo "--- Task: ${TASK} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=polaris_admet \
        ++constants.seed=${SEED} \
        ++constants.task=${TASK} \
        ++finetuning.task=${TASK} \
        ++datamodule.args.polaris_benchmark_names=${TASK} \
        +finetuning=polaris_admet \
        ++finetuning.pretrained_model=${CKPT} \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=none \
        ++finetuning.finetuning_head.in_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
        ++finetuning.new_out_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.depth=${ADDED_DEPTH} \
        ++finetuning.added_depth=${ADDED_DEPTH} \
        ++architecture.task_heads.${TASK}.hidden_dims=${FINETUNE_DIM} \
        ++trainer.model_checkpoint.dirpath=models_checkpoints/polaris_admet_test/${MODEL}/${TASK}/ \
        ++constants.max_epochs=5 \
        ++constants.wandb.entity=eddy26 \
        ++constants.wandb.project=graphium \
        "++constants.wandb.tags=['${MODEL}','polaris_admet_test']" \
        ++constants.results_csv_dir=results \
    || FAILED_TASKS+=("${TASK}")

    sleep 1
done

echo ""
echo "========================================="
if [[ ${#FAILED_TASKS[@]} -eq 0 ]]; then
    echo "=== ALL 6 TASKS PASSED ==="
else
    echo "=== FAILED TASKS (${#FAILED_TASKS[@]}/6): ${FAILED_TASKS[*]} ==="
fi
echo "Check results/experiment_results.csv for polaris metrics (pearsonr, mae, etc.)"
