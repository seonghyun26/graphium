#!/usr/bin/env bash
# Systematic caco2_wang fine-tuning sweep on GPU 7 from toymix_dti GPS++ 800M checkpoint.
# Goal: push MAE toward SOTA (~0.28-0.35).

set -euo pipefail
cd "$(dirname "$0")/.."

# Activate conda env for graphium-train
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate graphium 2>/dev/null || true
export PATH="/home/shpark/.conda/envs/graphium/bin:$PATH"

source "$(dirname "$0")/common.sh"

MODEL=gpspp_800M
CKPT="./models_checkpoints/toymix-dti/gpspp_800M/2026-03-21_22-57-10_20260321_225710/last.ckpt"
DEVICE=7
TASK=caco2_wang
FINETUNING_CONFIG=admet_toymix_dti
PRETRAIN_DATASET=toymix_dti

LOG_DIR="logs/caco2_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

run_experiment() {
    local name="$1"
    shift
    local extra_flags=("$@")

    echo "=========================================="
    echo "  Experiment: ${name}"
    echo "  Time: $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=admet \
        $(wandb_flags "['${MODEL}','finetune','admet','${PRETRAIN_DATASET}','caco2_sweep']") \
        ++constants.task=${TASK} \
        ++finetuning.task=${TASK} \
        ++datamodule.args.tdc_benchmark_names=${TASK} \
        +finetuning=${FINETUNING_CONFIG} \
        ++finetuning.pretrained_model=${CKPT} \
        ++trainer.model_checkpoint.dirpath=models_checkpoints/caco2_sweep/${name}/ \
        "${extra_flags[@]}" \
        2>&1 | tee "${LOG_DIR}/${name}.log"

    echo ">>> ${name} done at $(date)" >> "${LOG_DIR}/summary.txt"
    # Extract MAE from log
    grep -oP 'graph_caco2_wang/mae/test["\s:]+\K[0-9.]+' "${LOG_DIR}/${name}.log" >> "${LOG_DIR}/summary.txt" 2>/dev/null || true
    echo "" >> "${LOG_DIR}/summary.txt"

    sleep 2
}

echo "Starting caco2_wang sweep at $(date)" | tee "${LOG_DIR}/summary.txt"

# ============================================================================
# PHASE 1: Unfreezing schedule (biggest lever - toymix_dti runs never unfroze)
# ============================================================================
echo "=== PHASE 1: Unfreezing schedule ===" | tee -a "${LOG_DIR}/summary.txt"

# 1a: epoch_unfreeze_all=50, default LR and dims (FINETUNE_DIM=256, ADDED_DEPTH=4)
run_experiment "p1_unfreeze50" \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=50 \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${TASK}.hidden_dims=256

# 1b: epoch_unfreeze_all=30 (unfreeze earlier)
run_experiment "p1_unfreeze30" \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=30 \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${TASK}.hidden_dims=256

# 1c: epoch_unfreeze_all=70 (unfreeze later)
run_experiment "p1_unfreeze70" \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=70 \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${TASK}.hidden_dims=256

# 1d: gradual unfreezing: unfreeze_pretrained_depth=4, then full at 50
run_experiment "p1_gradual_d4_e50" \
    ++finetuning.unfreeze_pretrained_depth=4 \
    ++finetuning.epoch_unfreeze_all=50 \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${TASK}.hidden_dims=256

# 1e: Never unfreeze but with lower LR (reproduce current best baseline)
run_experiment "p1_frozen_baseline" \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=none \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${TASK}.hidden_dims=256

echo "=== PHASE 1 COMPLETE ===" | tee -a "${LOG_DIR}/summary.txt"

# ============================================================================
# PHASE 2: Learning rate sweep (with best unfreeze schedule from Phase 1)
# Use epoch_unfreeze_all=50 as default (best from overall results)
# ============================================================================
echo "=== PHASE 2: Learning rate sweep ===" | tee -a "${LOG_DIR}/summary.txt"

for lr in 1e-5 5e-5 1e-4 2e-4 5e-4; do
    run_experiment "p2_lr${lr}_unfreeze50" \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++predictor.optim_kwargs.lr=${lr} \
        ++finetuning.finetuning_head.in_dim=256 \
        ++finetuning.finetuning_head.hidden_dims=256 \
        ++finetuning.new_out_dim=256 \
        ++finetuning.finetuning_head.depth=4 \
        ++finetuning.added_depth=4 \
        ++architecture.task_heads.${TASK}.hidden_dims=256
done

# Also try LR with frozen backbone
for lr in 1e-4 5e-4 1e-3; do
    run_experiment "p2_lr${lr}_frozen" \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=none \
        ++predictor.optim_kwargs.lr=${lr} \
        ++finetuning.finetuning_head.in_dim=256 \
        ++finetuning.finetuning_head.hidden_dims=256 \
        ++finetuning.new_out_dim=256 \
        ++finetuning.finetuning_head.depth=4 \
        ++finetuning.added_depth=4 \
        ++architecture.task_heads.${TASK}.hidden_dims=256
done

echo "=== PHASE 2 COMPLETE ===" | tee -a "${LOG_DIR}/summary.txt"

# ============================================================================
# PHASE 3: Head architecture sweep (with best LR + unfreeze from P1/P2)
# ============================================================================
echo "=== PHASE 3: Head architecture ===" | tee -a "${LOG_DIR}/summary.txt"

# 3a: FINETUNE_DIM sweep
for dim in 64 128 512 1024; do
    run_experiment "p3_dim${dim}_unfreeze50" \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++finetuning.finetuning_head.in_dim=${dim} \
        ++finetuning.finetuning_head.hidden_dims=${dim} \
        ++finetuning.new_out_dim=${dim} \
        ++finetuning.finetuning_head.depth=4 \
        ++finetuning.added_depth=4 \
        ++architecture.task_heads.${TASK}.hidden_dims=${dim}
done

# 3b: ADDED_DEPTH sweep
for depth in 2 3 6; do
    run_experiment "p3_depth${depth}_unfreeze50" \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++finetuning.finetuning_head.in_dim=256 \
        ++finetuning.finetuning_head.hidden_dims=256 \
        ++finetuning.new_out_dim=256 \
        ++finetuning.finetuning_head.depth=${depth} \
        ++finetuning.added_depth=${depth} \
        ++architecture.task_heads.${TASK}.hidden_dims=256
done

# 3c: Dropout sweep
for dropout in 0.1 0.2 0.3; do
    run_experiment "p3_dropout${dropout}_unfreeze50" \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++finetuning.finetuning_head.in_dim=256 \
        ++finetuning.finetuning_head.hidden_dims=256 \
        ++finetuning.new_out_dim=256 \
        ++finetuning.finetuning_head.depth=4 \
        ++finetuning.added_depth=4 \
        ++architecture.task_heads.${TASK}.hidden_dims=256 \
        ++architecture.task_heads.${TASK}.dropout=${dropout}
done

echo "=== PHASE 3 COMPLETE ===" | tee -a "${LOG_DIR}/summary.txt"

# ============================================================================
# PHASE 4: Loss function and training schedule
# ============================================================================
echo "=== PHASE 4: Loss + schedule ===" | tee -a "${LOG_DIR}/summary.txt"

# 4a: MSE loss
run_experiment "p4_mse_unfreeze50" \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=50 \
    ++predictor.loss_fun.${TASK}=mse \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${TASK}.hidden_dims=256

# 4b: MSE loss with lower LR (MSE gradients are larger)
run_experiment "p4_mse_lr5e5_unfreeze50" \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=50 \
    ++predictor.loss_fun.${TASK}=mse \
    ++predictor.optim_kwargs.lr=5e-5 \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${TASK}.hidden_dims=256

# 4c: Extended training 200 epochs
run_experiment "p4_200ep_unfreeze100" \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=100 \
    ++constants.max_epochs=200 \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${TASK}.hidden_dims=256

# 4d: Extended training 300 epochs
run_experiment "p4_300ep_unfreeze150" \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=150 \
    ++constants.max_epochs=300 \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${TASK}.hidden_dims=256

echo "=== PHASE 4 COMPLETE ===" | tee -a "${LOG_DIR}/summary.txt"

# ============================================================================
# PHASE 5: Seed sweep on best config (will use unfreeze50 as default)
# ============================================================================
echo "=== PHASE 5: Seed sweep ===" | tee -a "${LOG_DIR}/summary.txt"

for seed in 1 2 3 42 123; do
    run_experiment "p5_seed${seed}_unfreeze50" \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++constants.seed=${seed} \
        ++finetuning.finetuning_head.in_dim=256 \
        ++finetuning.finetuning_head.hidden_dims=256 \
        ++finetuning.new_out_dim=256 \
        ++finetuning.finetuning_head.depth=4 \
        ++finetuning.added_depth=4 \
        ++architecture.task_heads.${TASK}.hidden_dims=256
done

echo "=== PHASE 5 COMPLETE ===" | tee -a "${LOG_DIR}/summary.txt"

echo ""
echo "============================================="
echo "  ALL PHASES COMPLETE at $(date)"
echo "  Results in: ${LOG_DIR}/summary.txt"
echo "  Full results in: results/experiment_results.csv"
echo "============================================="
