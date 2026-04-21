#!/usr/bin/env bash
# Fine-tune a pre-trained model on the 6 Polaris biogen/adme-fang-v1 ADME endpoints.
#
# Usage:
#   bash scripts/00_finetune_polaris_admet.sh <model> <checkpoint> [gpu_id]
#
# Arguments:
#   model      : gcn | mpnn | gpspp | gpspp_800M | pairformer | pairmixer_* ...
#   checkpoint : path to pre-trained .ckpt file
#   gpu_id     : CUDA device index (default: 5, matching our other scripts)
#
# Environment variables (optional):
#   FINETUNE_DIM         : hidden dim for finetuning head (default: 256)
#   UNFREEZE_DEPTH       : layers to unfreeze initially (default: 0, frozen backbone)
#   EPOCH_UNFREEZE_ALL   : epoch to unfreeze all layers (default: none, stay frozen)
#   ADDED_DEPTH          : depth of the appended finetuning head (default: 4)
#   FINETUNING_CONFIG    : hydra finetuning config (default: auto from PRETRAIN_DATASET)
#   SUB_MODULE           : sub_module_from_pretrained override
#   PRETRAIN_DATASET     : pretrain dataset name for W&B tags / config selection
#                          (default: auto-detect from ckpt path)
#
# Example:
#   bash scripts/00_finetune_polaris_admet.sh gpspp ./ckpt/gpspp_largemix.ckpt 5

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> <checkpoint> [gpu_id]"}
CKPT=${2:?   "Usage: $0 <model> <checkpoint> [gpu_id]"}
DEVICE=${3:-${DEVICE}}

FINETUNE_DIM=${FINETUNE_DIM:-256}
UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}
ADDED_DEPTH=${ADDED_DEPTH:-4}

# ── Infer pretrain dataset from checkpoint path (same logic as TDC ADMET) ─────
if [[ -z "${PRETRAIN_DATASET:-}" ]]; then
    CKPT_LOWER=$(echo "${CKPT}" | tr '[:upper:]' '[:lower:]')
    if [[ "${CKPT_LOWER}" == *"toymix_rxrx3_dti"* || "${CKPT_LOWER}" == *"toymix-rxrx3-dti"* ]]; then
        PRETRAIN_DATASET="toymix_rxrx3_dti"
    elif [[ "${CKPT_LOWER}" == *"toymix_rxrx3"* || "${CKPT_LOWER}" == *"toymix-rxrx3"* ]]; then
        PRETRAIN_DATASET="toymix_rxrx3"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti_esmc_v2"* || "${CKPT_LOWER}" == *"toymix-dti-esmc-v2"* ]]; then
        PRETRAIN_DATASET="toymix_dti_esmc_v2"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti_esmc"* || "${CKPT_LOWER}" == *"toymix-dti-esmc"* ]]; then
        PRETRAIN_DATASET="toymix_dti_esmc"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti_pactivity"* || "${CKPT_LOWER}" == *"toymix-dti-pactivity"* ]]; then
        PRETRAIN_DATASET="toymix_dti_pactivity"
    elif [[ "${CKPT_LOWER}" == *"dti_pactivity"* || "${CKPT_LOWER}" == *"dti-pactivity"* ]]; then
        PRETRAIN_DATASET="dti_pactivity"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti_10k_filtered"* || "${CKPT_LOWER}" == *"toymix-dti-10k-filtered"* ]]; then
        PRETRAIN_DATASET="toymix_dti_10k_filtered"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti_filtered"* || "${CKPT_LOWER}" == *"toymix-dti-filtered"* ]]; then
        PRETRAIN_DATASET="toymix_dti_filtered"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti_v2"* || "${CKPT_LOWER}" == *"toymix-dti-v2"* ]]; then
        PRETRAIN_DATASET="toymix_dti_v2"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti"* || "${CKPT_LOWER}" == *"toymix-dti"* ]]; then
        PRETRAIN_DATASET="toymix_dti"
    elif [[ "${CKPT_LOWER}" == *"toymix_lpm24_galactica"* || "${CKPT_LOWER}" == *"toymix-lpm24-galactica"* ]]; then
        PRETRAIN_DATASET="toymix_lpm24_galactica"
    elif [[ "${CKPT_LOWER}" == *"toymix_lpm24"* || "${CKPT_LOWER}" == *"toymix-lpm24"* ]]; then
        PRETRAIN_DATASET="toymix_lpm24"
    elif [[ "${CKPT_LOWER}" == *"rxrx3_dti"* || "${CKPT_LOWER}" == *"rxrx3-dti"* ]]; then
        PRETRAIN_DATASET="rxrx3_dti"
    elif [[ "${CKPT_LOWER}" == *"largemix_rxrx3_dti"* || "${CKPT_LOWER}" == *"largemix-rxrx3-dti"* ]]; then
        PRETRAIN_DATASET="largemix_rxrx3_dti"
    elif [[ "${CKPT_LOWER}" == *"largemix_rxrx3"* || "${CKPT_LOWER}" == *"largemix-rxrx3"* ]]; then
        PRETRAIN_DATASET="largemix_rxrx3"
    elif [[ "${CKPT_LOWER}" == *"largemix_dti_filtered"* || "${CKPT_LOWER}" == *"largemix-dti-filtered"* ]]; then
        PRETRAIN_DATASET="largemix_dti_filtered"
    elif [[ "${CKPT_LOWER}" == *"largemix_dti"* || "${CKPT_LOWER}" == *"largemix-dti"* ]]; then
        PRETRAIN_DATASET="largemix_dti"
    elif [[ "${CKPT_LOWER}" == *"largemix"* || "${CKPT_LOWER}" == *"large-dataset"* ]]; then
        PRETRAIN_DATASET="largemix"
    elif [[ "${CKPT_LOWER}" == *"toymix_bbbc047_filtered"* || "${CKPT_LOWER}" == *"toymix-bbbc047-filtered"* ]]; then
        PRETRAIN_DATASET="toymix_bbbc047_filtered"
    elif [[ "${CKPT_LOWER}" == *"toymix_bbbc047"* || "${CKPT_LOWER}" == *"toymix-bbbc047"* ]]; then
        PRETRAIN_DATASET="toymix_bbbc047"
    elif [[ "${CKPT_LOWER}" == *"toymix"* || "${CKPT_LOWER}" == *"small-dataset"* ]]; then
        PRETRAIN_DATASET="toymix"
    elif [[ "${CKPT_LOWER}" == *"rxrx3"* ]]; then
        PRETRAIN_DATASET="rxrx3"
    elif [[ "${CKPT_LOWER}" == *"dti"* ]]; then
        PRETRAIN_DATASET="dti"
    else
        PRETRAIN_DATASET="unknown"
    fi
fi

# ── Auto-select finetuning config (toymix → zinc, largemix → pcba_1328) ───────
if [[ -z "${FINETUNING_CONFIG:-}" ]]; then
    case "${PRETRAIN_DATASET}" in
        largemix) FINETUNING_CONFIG="polaris_admet_largemix" ;;
        *)        FINETUNING_CONFIG="polaris_admet" ;;
    esac
fi

TAGS="['${MODEL}','finetune','polaris_admet','${PRETRAIN_DATASET}'${MODEL_TAG:+,'${MODEL_TAG}'}]"

echo "=== Fine-tuning ${MODEL} on Polaris ADME-Fang (ckpt=${CKPT}, pretrain=${PRETRAIN_DATASET}, unfreeze=${UNFREEZE_DEPTH}) ==="

for task in "${POLARIS_ADME_TASKS[@]}"; do
    echo "--- Task: ${task} ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=polaris_admet \
        $(wandb_flags "${TAGS}") \
        ++constants.task=${task} \
        ++finetuning.task=${task} \
        ++datamodule.args.polaris_benchmark_names=${task} \
        +finetuning=${FINETUNING_CONFIG} \
        ++finetuning.pretrained_model=${CKPT} \
        ++finetuning.unfreeze_pretrained_depth=${UNFREEZE_DEPTH} \
        ++finetuning.epoch_unfreeze_all=${EPOCH_UNFREEZE_ALL} \
        ++finetuning.finetuning_head.in_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
        ++finetuning.new_out_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.depth=${ADDED_DEPTH} \
        ++finetuning.added_depth=${ADDED_DEPTH} \
        ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
        ++trainer.model_checkpoint.dirpath=models_checkpoints/polaris_admet/${PRETRAIN_DATASET}/${MODEL}/${task}/ \
        ${SUB_MODULE:+++finetuning.sub_module_from_pretrained=${SUB_MODULE}} \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
