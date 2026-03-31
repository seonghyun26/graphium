#!/usr/bin/env bash
# Fine-tune the GPS++ MoE pre-trained checkpoint on 5 representative ADMET tasks
# (one per category: Absorption, Distribution, Metabolism, Excretion, Toxicity).
#
# Usage:
#   bash scripts/04_finetune_moe_admet.sh [gpu_id]

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}

# ── MoE model config (must match pre-training) ─────────────────────────────
MODEL=gpspp
DIM=768
GNN_DEPTH=8

FINETUNE_DIM=${FINETUNE_DIM:-256}
ADDED_DEPTH=${ADDED_DEPTH:-4}
UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}

MOE_EXPERTS=4
MOE_TOP_K=2
MOE_AUX_COEFF=0.01

# ── Checkpoint ──────────────────────────────────────────────────────────────
CKPT="models_checkpoints/toymix-dti/gpspp_moe/toymix_dti_gpspp_moe_20260325_17020_20260325_170220/last.ckpt"
PRETRAIN_DATASET="toymix_dti_moe"
FINETUNING_CONFIG="admet_toymix_dti"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi

TAGS="['${MODEL}','moe','finetune','admet','${PRETRAIN_DATASET}']"

# 5 representative tasks: one per ADMET category
MOE_TASKS=(
    caco2_wang              # Absorption (regression, MAE)
    bbb_martins             # Distribution (classification, AUROC)
    cyp3a4_veith            # Metabolism (classification, AUPRC)
    half_life_obach         # Excretion (regression, Spearman)
    herg                    # Toxicity (classification, AUROC)
)

echo "=== Fine-tuning GPS++ MoE on 5 ADMET tasks (one per category) ==="
echo "  Checkpoint: ${CKPT}"
echo "  Model: ${MODEL} (dim=${DIM}, depth=${GNN_DEPTH}, MoE=${MOE_EXPERTS}exp/top${MOE_TOP_K})"
echo "  Tasks: ${MOE_TASKS[*]}"
echo "  GPU: ${DEVICE}"
echo ""

for task in "${MOE_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

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
        ++architecture.gnn.depth=${GNN_DEPTH} \
        $(gpspp_dim_flags "${DIM}") \
        ++architecture.gnn.layer_kwargs.moe_num_experts=${MOE_EXPERTS} \
        ++architecture.gnn.layer_kwargs.moe_top_k=${MOE_TOP_K} \
        ++architecture.gnn.layer_kwargs.moe_aux_loss_coeff=${MOE_AUX_COEFF} \
        ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
        ++trainer.model_checkpoint.save_last=False \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done

echo ""
echo "Done. 5 ADMET tasks completed."
echo "Results tagged with 'moe' in: results/experiment_results.csv"
