#!/usr/bin/env bash
# Round 3: Test gnn_layer_pooling (multi-fingerprint probing) for caco2_wang.
# Uses the best settings from sweep 1+2 combined with layer pooling.

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

LOG_DIR="logs/caco2_sweep3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# Best config from sweep 1+2: gradual unfreezing d4+e50, dim128, depth2, dropout0.2
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
)

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
        $(wandb_flags "['${MODEL}','finetune','admet','${PRETRAIN_DATASET}','caco2_sweep3_layerpool']") \
        ++constants.task=${TASK} \
        ++finetuning.task=${TASK} \
        ++datamodule.args.tdc_benchmark_names=${TASK} \
        +finetuning=${FINETUNING_CONFIG} \
        ++finetuning.pretrained_model=${CKPT} \
        ++trainer.model_checkpoint.dirpath=models_checkpoints/caco2_sweep3/${name}/ \
        "${BASE_FLAGS[@]}" \
        "${extra_flags[@]}" \
        2>&1 | tee "${LOG_DIR}/${name}.log"

    echo ">>> ${name} done at $(date)" >> "${LOG_DIR}/summary.txt"
    sleep 2
}

echo "Starting caco2_wang sweep3 (gnn_layer_pooling) at $(date)" | tee "${LOG_DIR}/summary.txt"

# ============================================================================
# BASELINE: best config without layer pooling (for comparison)
# ============================================================================
for seed in 0 1 2; do
    run_experiment "baseline_seed${seed}" \
        ++constants.seed=${seed}
done

# ============================================================================
# WEIGHTED SUM: combine different layer subsets
# GPS++ 800M has 12 layers (indices 0-11)
# ============================================================================

# All layers
for seed in 0 1 2; do
    run_experiment "ws_all_seed${seed}" \
        ++constants.seed=${seed} \
        '++architecture.gnn_layer_pooling.layers=[0,1,2,3,4,5,6,7,8,9,10,11]' \
        ++architecture.gnn_layer_pooling.mode=weighted_sum
done

# Last 4 layers (8,9,10,11) - similar to BERT top-4
for seed in 0 1 2; do
    run_experiment "ws_last4_seed${seed}" \
        ++constants.seed=${seed} \
        '++architecture.gnn_layer_pooling.layers=[8,9,10,11]' \
        ++architecture.gnn_layer_pooling.mode=weighted_sum
done

# Evenly spaced (0, 3, 7, 11) - captures low/mid/high features
for seed in 0 1 2; do
    run_experiment "ws_spaced_seed${seed}" \
        ++constants.seed=${seed} \
        '++architecture.gnn_layer_pooling.layers=[0,3,7,11]' \
        ++architecture.gnn_layer_pooling.mode=weighted_sum
done

# Early + late (0, 1, 10, 11)
for seed in 0 1 2; do
    run_experiment "ws_early_late_seed${seed}" \
        ++constants.seed=${seed} \
        '++architecture.gnn_layer_pooling.layers=[0,1,10,11]' \
        ++architecture.gnn_layer_pooling.mode=weighted_sum
done

# ============================================================================
# CONCAT_PROJ: concatenation + linear projection
# ============================================================================

# Evenly spaced (0, 3, 7, 11)
for seed in 0 1 2; do
    run_experiment "cp_spaced_seed${seed}" \
        ++constants.seed=${seed} \
        '++architecture.gnn_layer_pooling.layers=[0,3,7,11]' \
        ++architecture.gnn_layer_pooling.mode=concat_proj
done

# Last 4 layers
for seed in 0 1 2; do
    run_experiment "cp_last4_seed${seed}" \
        ++constants.seed=${seed} \
        '++architecture.gnn_layer_pooling.layers=[8,9,10,11]' \
        ++architecture.gnn_layer_pooling.mode=concat_proj
done

# ============================================================================
# LAYER POOLING + FROZEN BACKBONE (pure probing, no fine-tuning)
# ============================================================================
for seed in 0 1 2; do
    run_experiment "ws_spaced_frozen_seed${seed}" \
        ++constants.seed=${seed} \
        '++architecture.gnn_layer_pooling.layers=[0,3,7,11]' \
        ++architecture.gnn_layer_pooling.mode=weighted_sum \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=none
done

echo ""
echo "============================================="
echo "  SWEEP 3 COMPLETE at $(date)"
echo "============================================="
