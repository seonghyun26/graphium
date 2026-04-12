#!/usr/bin/env bash
# Verification experiments on GPU 4:
#   1. Reproduce best config (same seed) - should hit ~0.29 MAE
#   2. Cosine annealing with non-zero eta_min
#   3. Reproduce best config with different seed

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

LOG_DIR="logs/caco2_verify_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# Best config discovered this session
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
)

run_experiment() {
    local name="$1"
    shift
    local extra_flags=("$@")

    echo ""
    echo "=========================================="
    echo "  Experiment: ${name}"
    echo "  Time: $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=admet \
        $(wandb_flags "['${MODEL}','finetune','admet','${PRETRAIN_DATASET}','caco2_verify','${name}']") \
        ++constants.task=${TASK} \
        ++finetuning.task=${TASK} \
        ++datamodule.args.tdc_benchmark_names=${TASK} \
        +finetuning=${FINETUNING_CONFIG} \
        ++finetuning.pretrained_model=${CKPT} \
        ++trainer.model_checkpoint.dirpath=models_checkpoints/caco2_verify/${name}/ \
        "${BASE_FLAGS[@]}" \
        "${extra_flags[@]}" \
        2>&1 | tee "${LOG_DIR}/${name}.log"

    echo ">>> ${name} done at $(date)" >> "${LOG_DIR}/summary.txt"
    sleep 2
}

echo "Starting caco2_verify at $(date)" | tee "${LOG_DIR}/summary.txt"

# 1) Reproduce best config with seed=0 (should reproduce MAE=0.293)
run_experiment "reproduce_seed0" \
    ++constants.seed=0

# 2) Cosine annealing with non-zero eta_min (keep some learning signal at the end)
# Use ~ to remove WarmUpLinearLR-specific kwargs before setting CosineAnnealingLR
run_experiment "cosine_annealing" \
    ++constants.seed=0 \
    ~predictor.torch_scheduler_kwargs.max_num_epochs \
    ~predictor.torch_scheduler_kwargs.warmup_epochs \
    ~predictor.torch_scheduler_kwargs.verbose \
    ++predictor.torch_scheduler_kwargs.module_type=CosineAnnealingLR \
    ++predictor.torch_scheduler_kwargs.T_max=200 \
    ++predictor.torch_scheduler_kwargs.eta_min=1e-6

# 3) Reproduce best config with different seed (test robustness)
run_experiment "reproduce_seed7" \
    ++constants.seed=7

echo "" | tee -a "${LOG_DIR}/summary.txt"
echo "============================================="
echo "  VERIFY DONE at $(date)"
echo "============================================="
