#!/usr/bin/env bash
# Cosine annealing experiments across seeds 1, 2 for fair comparison with constant LR

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate graphium 2>/dev/null || true
export PATH="/home/shpark/.conda/envs/graphium/bin:$PATH"

source "$(dirname "$0")/common.sh"

MODEL=gpspp_800M
CKPT="./models_checkpoints/toymix-dti/gpspp_800M/2026-03-21_22-57-10_20260321_225710/last.ckpt"
DEVICE=4
TASK=caco2_wang
FINETUNING_CONFIG=admet_toymix_dti
PRETRAIN_DATASET=toymix_dti

LOG_DIR="logs/caco2_cosine_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

BASE_FLAGS=(
    ++finetuning.unfreeze_pretrained_depth=4
    ++finetuning.epoch_unfreeze_all=50
    ++finetuning.finetuning_head.in_dim=128
    ++finetuning.finetuning_head.hidden_dims=128
    ++finetuning.new_out_dim=128
    ++finetuning.finetuning_head.depth=2
    ++finetuning.added_depth=2
    ++architecture.task_heads.${TASK}.hidden_dims=128
    ++architecture.task_heads.${TASK}.dropout=0.2
    ++constants.max_epochs=200
    ++datamodule.args.num_workers=0
    ++datamodule.args.featurization_n_jobs=0
    ~predictor.torch_scheduler_kwargs.max_num_epochs
    ~predictor.torch_scheduler_kwargs.warmup_epochs
    ~predictor.torch_scheduler_kwargs.verbose
    ++predictor.torch_scheduler_kwargs.module_type=CosineAnnealingLR
    ++predictor.torch_scheduler_kwargs.T_max=200
    ++predictor.torch_scheduler_kwargs.eta_min=1e-6
)

for seed in 1 2; do
    echo ""
    echo "=========================================="
    echo "  cosine_seed${seed} @ $(date)"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=admet \
        $(wandb_flags "['${MODEL}','finetune','admet','${PRETRAIN_DATASET}','caco2_cosine_seeds']") \
        ++constants.task=${TASK} \
        ++finetuning.task=${TASK} \
        ++datamodule.args.tdc_benchmark_names=${TASK} \
        +finetuning=${FINETUNING_CONFIG} \
        ++finetuning.pretrained_model=${CKPT} \
        ++trainer.model_checkpoint.dirpath=models_checkpoints/caco2_cosine/seed${seed}/ \
        ++constants.seed=${seed} \
        "${BASE_FLAGS[@]}" \
        2>&1 | tee "${LOG_DIR}/cosine_seed${seed}.log"
done

echo "DONE at $(date)"
