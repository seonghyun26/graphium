#!/usr/bin/env bash
# Round 2: Combine winning factors from Phase 1 sweep.
# Winners: gradual unfreezing (d4+e50), shallow heads (depth 2-3), dim 128, dropout 0.2
# Target: push MAE below 0.30

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate graphium 2>/dev/null || true
export PATH="/home/shpark/.conda/envs/graphium/bin:$PATH"

source "$(dirname "$0")/common.sh"

MODEL=gpspp_800M
CKPT="./models_checkpoints/toymix-dti/gpspp_800M/2026-03-21_22-57-10_20260321_225710/last.ckpt"
DEVICE=7
TASK=caco2_wang
FINETUNING_CONFIG=admet_toymix_dti
PRETRAIN_DATASET=toymix_dti

LOG_DIR="logs/caco2_sweep2_$(date +%Y%m%d_%H%M%S)"
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
        $(wandb_flags "['${MODEL}','finetune','admet','${PRETRAIN_DATASET}','caco2_sweep2']") \
        ++constants.task=${TASK} \
        ++finetuning.task=${TASK} \
        ++datamodule.args.tdc_benchmark_names=${TASK} \
        +finetuning=${FINETUNING_CONFIG} \
        ++finetuning.pretrained_model=${CKPT} \
        ++trainer.model_checkpoint.dirpath=models_checkpoints/caco2_sweep2/${name}/ \
        "${extra_flags[@]}" \
        2>&1 | tee "${LOG_DIR}/${name}.log"

    echo ">>> ${name} done at $(date)" >> "${LOG_DIR}/summary.txt"
    sleep 2
}

echo "Starting caco2_wang sweep2 at $(date)" | tee "${LOG_DIR}/summary.txt"

# ============================================================================
# COMBINED CONFIGS: gradual unfreezing + shallow head + smaller dim + dropout
# ============================================================================

# --- Best combo: gradual(d4) + depth2 + dim128 + dropout0.2 ---
for seed in 0 1 2 3 42 123; do
    run_experiment "combo_best_seed${seed}" \
        ++finetuning.unfreeze_pretrained_depth=4 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++constants.seed=${seed} \
        ++finetuning.finetuning_head.in_dim=128 \
        ++finetuning.finetuning_head.hidden_dims=128 \
        ++finetuning.new_out_dim=128 \
        ++finetuning.finetuning_head.depth=2 \
        ++finetuning.added_depth=2 \
        ++architecture.task_heads.${TASK}.hidden_dims=128 \
        ++architecture.task_heads.${TASK}.dropout=0.2
done

# --- Variant: gradual(d4) + depth3 + dim128 + dropout0.2 ---
for seed in 0 1 2 3 42 123; do
    run_experiment "combo_d3_seed${seed}" \
        ++finetuning.unfreeze_pretrained_depth=4 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++constants.seed=${seed} \
        ++finetuning.finetuning_head.in_dim=128 \
        ++finetuning.finetuning_head.hidden_dims=128 \
        ++finetuning.new_out_dim=128 \
        ++finetuning.finetuning_head.depth=3 \
        ++finetuning.added_depth=3 \
        ++architecture.task_heads.${TASK}.hidden_dims=128 \
        ++architecture.task_heads.${TASK}.dropout=0.2
done

# --- Variant: gradual(d4) + depth2 + dim256 + dropout0.2 (original dim, shallower head) ---
for seed in 0 1 2 3 42 123; do
    run_experiment "combo_dim256_d2_seed${seed}" \
        ++finetuning.unfreeze_pretrained_depth=4 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++constants.seed=${seed} \
        ++finetuning.finetuning_head.in_dim=256 \
        ++finetuning.finetuning_head.hidden_dims=256 \
        ++finetuning.new_out_dim=256 \
        ++finetuning.finetuning_head.depth=2 \
        ++finetuning.added_depth=2 \
        ++architecture.task_heads.${TASK}.hidden_dims=256 \
        ++architecture.task_heads.${TASK}.dropout=0.2
done

# --- Extended training: 200 epochs with gradual unfreezing ---
for seed in 0 1 2 42; do
    run_experiment "combo_200ep_seed${seed}" \
        ++finetuning.unfreeze_pretrained_depth=4 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++constants.max_epochs=200 \
        ++constants.seed=${seed} \
        ++finetuning.finetuning_head.in_dim=128 \
        ++finetuning.finetuning_head.hidden_dims=128 \
        ++finetuning.new_out_dim=128 \
        ++finetuning.finetuning_head.depth=2 \
        ++finetuning.added_depth=2 \
        ++architecture.task_heads.${TASK}.hidden_dims=128 \
        ++architecture.task_heads.${TASK}.dropout=0.2
done

# --- Deeper gradual unfreezing: depth=6 initially ---
for seed in 0 1 2 42; do
    run_experiment "combo_d6_seed${seed}" \
        ++finetuning.unfreeze_pretrained_depth=6 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++constants.seed=${seed} \
        ++finetuning.finetuning_head.in_dim=128 \
        ++finetuning.finetuning_head.hidden_dims=128 \
        ++finetuning.new_out_dim=128 \
        ++finetuning.finetuning_head.depth=2 \
        ++finetuning.added_depth=2 \
        ++architecture.task_heads.${TASK}.hidden_dims=128 \
        ++architecture.task_heads.${TASK}.dropout=0.2
done

# --- Earlier full unfreezing: epoch_unfreeze_all=30 ---
for seed in 0 1 2 42; do
    run_experiment "combo_e30_seed${seed}" \
        ++finetuning.unfreeze_pretrained_depth=4 \
        ++finetuning.epoch_unfreeze_all=30 \
        ++constants.seed=${seed} \
        ++finetuning.finetuning_head.in_dim=128 \
        ++finetuning.finetuning_head.hidden_dims=128 \
        ++finetuning.new_out_dim=128 \
        ++finetuning.finetuning_head.depth=2 \
        ++finetuning.added_depth=2 \
        ++architecture.task_heads.${TASK}.hidden_dims=128 \
        ++architecture.task_heads.${TASK}.dropout=0.2
done

# --- LR variants with best combo ---
for lr in 5e-5 2e-4; do
    for seed in 0 1 2; do
        run_experiment "combo_lr${lr}_seed${seed}" \
            ++finetuning.unfreeze_pretrained_depth=4 \
            ++finetuning.epoch_unfreeze_all=50 \
            ++predictor.optim_kwargs.lr=${lr} \
            ++constants.seed=${seed} \
            ++finetuning.finetuning_head.in_dim=128 \
            ++finetuning.finetuning_head.hidden_dims=128 \
            ++finetuning.new_out_dim=128 \
            ++finetuning.finetuning_head.depth=2 \
            ++finetuning.added_depth=2 \
            ++architecture.task_heads.${TASK}.hidden_dims=128 \
            ++architecture.task_heads.${TASK}.dropout=0.2
    done
done

# --- Dropout 0.1 with gradual unfreezing ---
for seed in 0 1 2 42; do
    run_experiment "combo_drop01_seed${seed}" \
        ++finetuning.unfreeze_pretrained_depth=4 \
        ++finetuning.epoch_unfreeze_all=50 \
        ++constants.seed=${seed} \
        ++finetuning.finetuning_head.in_dim=128 \
        ++finetuning.finetuning_head.hidden_dims=128 \
        ++finetuning.new_out_dim=128 \
        ++finetuning.finetuning_head.depth=2 \
        ++finetuning.added_depth=2 \
        ++architecture.task_heads.${TASK}.hidden_dims=128 \
        ++architecture.task_heads.${TASK}.dropout=0.1
done

echo ""
echo "============================================="
echo "  SWEEP 2 COMPLETE at $(date)"
echo "============================================="
