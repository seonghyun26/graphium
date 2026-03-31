#!/usr/bin/env bash
# Pilot test: GPS++ 800M with MoE on toymix+dti, then finetune on one ADMET task.
#
# Usage:
#   bash scripts/08_pilot_moe_gpspp.sh [gpu_id]
#
# This runs:
#   1. Pre-train GPS++ with MoE (4 experts, top-2) on toymix_dti for 50 epochs
#   2. Fine-tune on caco2_wang (regression ADMET task)
#   3. Compare against a baseline checkpoint (no MoE, if available)

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}
MODEL=gpspp
DATASET=toymix_dti
FT_TASK=caco2_wang

# ── Pre-training config ──────────────────────────────────────────────────────
# Use a smaller GPS++ to fit on one GPU with MoE overhead
DIM=768
GNN_DEPTH=8
BATCH_SIZE=128
MAX_EPOCHS=50

MOE_EXPERTS=4
MOE_TOP_K=2
MOE_AUX_COEFF=0.01

CKPT_DIR="models_checkpoints/toymix-dti/gpspp_moe/${DATASET}_gpspp_moe_$(date +%Y%m%d_%H%M%S)"
TAGS="['gpspp','moe','pretrain','${DATASET}']"

echo "=== Pilot: GPS++ MoE (${MOE_EXPERTS} experts, top-${MOE_TOP_K}) on ${DATASET} ==="
echo "Config: dim=${DIM}, depth=${GNN_DEPTH}, bs=${BATCH_SIZE}, epochs=${MAX_EPOCHS}"
echo "GPU: ${DEVICE}"
echo ""

# ── Phase 1: Pre-train with MoE ─────────────────────────────────────────────
echo "--- Phase 1: Pre-training on ${DATASET} ---"

CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
    model=${MODEL} \
    accelerator=gpu \
    tasks=${DATASET} \
    training=${DATASET} \
    architecture=toymix \
    $(wandb_flags "${TAGS}") \
    ++constants.max_epochs=${MAX_EPOCHS} \
    ++constants.raise_train_error=true \
    ++architecture.gnn.depth=${GNN_DEPTH} \
    $(gpspp_dim_flags "${DIM}") \
    ++datamodule.args.batch_size_training=${BATCH_SIZE} \
    ++architecture.gnn.layer_kwargs.moe_num_experts=${MOE_EXPERTS} \
    ++architecture.gnn.layer_kwargs.moe_top_k=${MOE_TOP_K} \
    ++architecture.gnn.layer_kwargs.moe_aux_loss_coeff=${MOE_AUX_COEFF} \
    ++trainer.trainer.precision=bf16-mixed \
    ++trainer.trainer.check_val_every_n_epoch=10 \
    ++trainer.model_checkpoint.dirpath=${CKPT_DIR} \
    ++trainer.model_checkpoint.save_last=True \
    ${EXTRA_FLAGS:-}

# Find the checkpoint
CKPT=$(find "${CKPT_DIR}" -name "last.ckpt" -type f 2>/dev/null | head -1)
if [[ -z "${CKPT}" ]]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo ""
echo "Checkpoint: ${CKPT}"

# ── Phase 2: Fine-tune on ADMET ─────────────────────────────────────────────
echo ""
echo "--- Phase 2: Fine-tuning on ${FT_TASK} ---"

FT_TAGS="['gpspp','moe','finetune','${FT_TASK}']"

CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
    model=${MODEL} \
    accelerator=gpu \
    tasks=admet \
    +finetuning=admet_toymix_dti \
    $(wandb_flags "${FT_TAGS}") \
    ++constants.raise_train_error=False \
    ++constants.task=${FT_TASK} \
    ++finetuning.task=${FT_TASK} \
    ++datamodule.args.tdc_benchmark_names=${FT_TASK} \
    ++datamodule.args.num_workers=0 \
    ++datamodule.args.featurization_n_jobs=0 \
    ++datamodule.args.processed_graph_data_path=null \
    ++finetuning.pretrained_model=${CKPT} \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=none \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${FT_TASK}.hidden_dims=256 \
    ++trainer.model_checkpoint.save_last=False \
    ${EXTRA_FLAGS:-}

echo ""
echo "=== Done: GPS++ MoE pilot test ==="
echo "Check wandb for results."
