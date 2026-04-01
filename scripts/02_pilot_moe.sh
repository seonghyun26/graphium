#!/usr/bin/env bash
# MoE pilot: pre-train GPS++ with Mixture of Experts and fine-tune on ADMET.
#
# Usage:
#   bash scripts/02_pilot_moe.sh pretrain_finetune [gpu_id]    # end-to-end: pretrain + finetune
#   bash scripts/02_pilot_moe.sh finetune <checkpoint> [gpu_id] # finetune existing MoE checkpoint
#
# Environment variables:
#   DIM, GNN_DEPTH, BATCH_SIZE, MOE_EXPERTS, MOE_TOP_K  — override defaults
#   FINETUNE_DIM, ADDED_DEPTH, UNFREEZE_DEPTH, EPOCH_UNFREEZE_ALL — finetuning config

source "$(dirname "$0")/common.sh"

MODE=${1:?  "Usage: $0 <pretrain_finetune|finetune> [checkpoint] [gpu_id]"}

# ── MoE model config ───────────────────────────────────────────────────────
MODEL=gpspp
DIM=${DIM:-768}
GNN_DEPTH=${GNN_DEPTH:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
MAX_EPOCHS=${MAX_EPOCHS:-50}

MOE_EXPERTS=${MOE_EXPERTS:-4}
MOE_TOP_K=${MOE_TOP_K:-2}
MOE_AUX_COEFF=${MOE_AUX_COEFF:-0.01}

MOE_FLAGS="\
    ++architecture.gnn.layer_kwargs.moe_num_experts=${MOE_EXPERTS} \
    ++architecture.gnn.layer_kwargs.moe_top_k=${MOE_TOP_K} \
    ++architecture.gnn.layer_kwargs.moe_aux_loss_coeff=${MOE_AUX_COEFF}"

# ── Finetuning config ──────────────────────────────────────────────────────
FINETUNE_DIM=${FINETUNE_DIM:-256}
ADDED_DEPTH=${ADDED_DEPTH:-4}
UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}

# 5 representative ADMET tasks (one per category)
MOE_TASKS=(
    caco2_wang              # Absorption (regression, MAE)
    bbb_martins             # Distribution (classification, AUROC)
    cyp3a4_veith            # Metabolism (classification, AUPRC)
    half_life_obach         # Excretion (regression, Spearman)
    herg                    # Toxicity (classification, AUROC)
)

# ── Helper: finetune MoE checkpoint on ADMET tasks ─────────────────────────
finetune_moe() {
    local CKPT=$1
    local DEVICE=$2
    local PRETRAIN_DATASET=${3:-toymix_dti_moe}
    local FINETUNING_CONFIG=${4:-admet_toymix_dti}

    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: Checkpoint not found: ${CKPT}"
        exit 1
    fi

    local TAGS="['${MODEL}','moe','${MOE_EXPERTS}exp_top${MOE_TOP_K}','finetune','admet','${PRETRAIN_DATASET}']"

    echo "=== Fine-tuning GPS++ MoE on ${#MOE_TASKS[@]} ADMET tasks ==="
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
            ++constants.task=${task} \
            ++finetuning.task=${task} \
            ++datamodule.args.tdc_benchmark_names=${task} \
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
            ${MOE_FLAGS} \
            ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
            ${EXTRA_FLAGS:-} \
        || echo "WARN: task ${task} failed, continuing..."

        sleep 1
    done

    echo ""
    echo "Done. ${#MOE_TASKS[@]} ADMET tasks completed."
}

# ── Main ────────────────────────────────────────────────────────────────────
case "${MODE}" in
    pretrain_finetune)
        DEVICE=${2:-${DEVICE}}
        DATASET=${DATASET:-toymix_dti}

        CKPT_DIR="models_checkpoints/toymix-dti/gpspp_moe/${DATASET}_gpspp_moe_$(date +%Y%m%d_%H%M%S)"
        TAGS="['gpspp','moe','pretrain','${DATASET}']"

        echo "=== Pilot: GPS++ MoE (${MOE_EXPERTS} experts, top-${MOE_TOP_K}) on ${DATASET} ==="
        echo "Config: dim=${DIM}, depth=${GNN_DEPTH}, bs=${BATCH_SIZE}, epochs=${MAX_EPOCHS}"
        echo "GPU: ${DEVICE}"
        echo ""

        # Phase 1: Pre-train
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
            ${MOE_FLAGS} \
            ++trainer.trainer.precision=bf16-mixed \
            ++trainer.trainer.check_val_every_n_epoch=10 \
            ++trainer.model_checkpoint.dirpath=${CKPT_DIR} \
            ++trainer.model_checkpoint.save_last=True \
            ${EXTRA_FLAGS:-}

        CKPT=$(find "${CKPT_DIR}" -name "last.ckpt" -type f 2>/dev/null | head -1)
        if [[ -z "${CKPT}" ]]; then
            echo "ERROR: No checkpoint found in ${CKPT_DIR}"
            exit 1
        fi
        echo "Checkpoint: ${CKPT}"

        # Phase 2: Fine-tune
        echo ""
        echo "--- Phase 2: Fine-tuning on ADMET ---"
        finetune_moe "${CKPT}" "${DEVICE}"
        ;;

    finetune)
        CKPT=${2:?  "Usage: $0 finetune <checkpoint> [gpu_id]"}
        DEVICE=${3:-${DEVICE}}
        finetune_moe "${CKPT}" "${DEVICE}"
        ;;

    *)
        echo "Usage: $0 <pretrain_finetune|finetune> [args...]"
        echo ""
        echo "Modes:"
        echo "  pretrain_finetune [gpu_id]          End-to-end: pretrain + finetune on 5 ADMET tasks"
        echo "  finetune <checkpoint> [gpu_id]      Finetune existing MoE checkpoint on 5 ADMET tasks"
        exit 1
        ;;
esac

echo ""
echo "=== Done: GPS++ MoE pilot ==="
