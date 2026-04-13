#!/usr/bin/env bash
# Fine-tune a pre-trained model on all 22 ADMET benchmark tasks.
#
# Usage:
#   bash scripts/00_finetune_admet.sh <model> <checkpoint> [gpu_id]
#
# Arguments:
#   model      : gcn | mpnn | gpspp | gpspp_800M
#   checkpoint : path to pre-trained .ckpt file
#   gpu_id     : CUDA device index (default: 0)
#
# Environment variables (optional):
#   FINETUNE_DIM         : hidden dim for finetuning head (default: 256)
#   UNFREEZE_DEPTH       : layers to unfreeze initially (default: 0, frozen backbone)
#   EPOCH_UNFREEZE_ALL   : epoch to unfreeze all layers (default: none, stay frozen)
#   FINETUNING_CONFIG    : hydra finetuning config (default: admet)
#   SUB_MODULE           : sub_module_from_pretrained (default: auto-detect)
#   PRETRAIN_DATASET     : pretrain dataset name for W&B tags (default: auto-detect from ckpt path)
#   USE_COSINE=1         : use the *_cosine variant (200 epochs + CosineAnnealingLR)
#                          appends "_cosine" to the auto-selected FINETUNING_CONFIG
#
# Examples:
#   bash scripts/00_finetune_admet.sh gpspp ./checkpoints/gpspp_largemix.ckpt 0
#   UNFREEZE_DEPTH=4 EPOCH_UNFREEZE_ALL=40 bash scripts/00_finetune_admet.sh gcn ./ckpt.ckpt 1

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> <checkpoint> [gpu_id]"}
CKPT=${2:?   "Usage: $0 <model> <checkpoint> [gpu_id]"}
DEVICE=${3:-${DEVICE}}

FINETUNE_DIM=${FINETUNE_DIM:-256}
UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}
ADDED_DEPTH=${ADDED_DEPTH:-4}

# ── Determine sub_module and dim flags based on model ────────────────────────
case "${MODEL}" in
    gcn)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    mpnn)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    gpspp)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    gpspp_800M)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    pairformer)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    pairformer_boltz)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    pairformer_17M|pairformer_52M)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    pairmixer_10M|pairmixer_20M|pairmixer_40M|pairmixer_boltz)
        DIM_FLAGS="++architecture.task_heads.\${task}.hidden_dims=${FINETUNE_DIM}"
        ;;
    *)
        echo "Error: unknown model '${MODEL}'."
        exit 1
        ;;
esac

# ── Infer pretrain dataset from checkpoint path for W&B tags ──────────────────
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
    elif [[ "${CKPT_LOWER}" == *"toymix_dti_10k_filtered"* || "${CKPT_LOWER}" == *"toymix-dti-10k-filtered"* ]]; then
        PRETRAIN_DATASET="toymix_dti_10k_filtered"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti_filtered"* || "${CKPT_LOWER}" == *"toymix-dti-filtered"* ]]; then
        PRETRAIN_DATASET="toymix_dti_filtered"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti_v2"* || "${CKPT_LOWER}" == *"toymix-dti-v2"* ]]; then
        PRETRAIN_DATASET="toymix_dti_v2"
    elif [[ "${CKPT_LOWER}" == *"toymix_dti"* || "${CKPT_LOWER}" == *"toymix-dti"* ]]; then
        PRETRAIN_DATASET="toymix_dti"
    elif [[ "${CKPT_LOWER}" == *"toymix_lpm24"* || "${CKPT_LOWER}" == *"toymix-lpm24"* ]]; then
        PRETRAIN_DATASET="toymix_lpm24"
    elif [[ "${CKPT_LOWER}" == *"rxrx3_dti"* || "${CKPT_LOWER}" == *"rxrx3-dti"* ]]; then
        PRETRAIN_DATASET="rxrx3_dti"
    elif [[ "${CKPT_LOWER}" == *"largemix_rxrx3"* || "${CKPT_LOWER}" == *"largemix-rxrx3"* ]]; then
        PRETRAIN_DATASET="largemix_rxrx3"
    elif [[ "${CKPT_LOWER}" == *"largemix_rxrx3_dti"* || "${CKPT_LOWER}" == *"largemix-rxrx3-dti"* ]]; then
        PRETRAIN_DATASET="largemix_rxrx3_dti"
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

# ── Auto-select finetuning config based on pretrain dataset ───────────────────
if [[ -z "${FINETUNING_CONFIG:-}" ]]; then
    case "${PRETRAIN_DATASET}" in
        toymix)         FINETUNING_CONFIG="admet" ;;                  # sub_module: zinc
        largemix)       FINETUNING_CONFIG="admet_largemix" ;;         # sub_module: pcba_1328
        rxrx3)          FINETUNING_CONFIG="admet_rxrx3" ;;            # sub_module: rxrx3
        largemix_rxrx3) FINETUNING_CONFIG="admet_largemix_rxrx3" ;;   # sub_module: rxrx3
        toymix_rxrx3)   FINETUNING_CONFIG="admet_toymix_rxrx3" ;;     # sub_module: rxrx3
        dti)            FINETUNING_CONFIG="admet_dti" ;;               # sub_module: dti
        largemix_dti)   FINETUNING_CONFIG="admet_largemix_dti" ;;      # sub_module: dti
        largemix_dti_filtered) FINETUNING_CONFIG="admet_largemix_dti_filtered" ;; # sub_module: dti
        toymix_dti)     FINETUNING_CONFIG="admet_toymix_dti" ;;        # sub_module: dti
        toymix_dti_v2)  FINETUNING_CONFIG="admet_toymix_dti" ;;       # sub_module: dti
        toymix_dti_esmc) FINETUNING_CONFIG="admet_toymix_dti" ;;      # sub_module: dti
        toymix_dti_esmc_v2) FINETUNING_CONFIG="admet_toymix_dti" ;;   # sub_module: dti
        toymix_dti_10k_filtered) FINETUNING_CONFIG="admet_toymix_dti_10k_filtered" ;; # sub_module: dti
        toymix_dti_filtered) FINETUNING_CONFIG="admet_toymix_dti_filtered" ;; # sub_module: dti
        toymix_lpm24)   FINETUNING_CONFIG="admet_toymix_lpm24" ;;     # sub_module: lpm24
        rxrx3_dti)      FINETUNING_CONFIG="admet_rxrx3_dti" ;;       # sub_module: dti
        toymix_rxrx3_dti) FINETUNING_CONFIG="admet_toymix_rxrx3_dti" ;;  # sub_module: dti
        largemix_rxrx3_dti) FINETUNING_CONFIG="admet_largemix_rxrx3_dti" ;; # sub_module: dti
        toymix_bbbc047) FINETUNING_CONFIG="admet_toymix_bbbc047" ;; # sub_module: bbbc047
        toymix_bbbc047_filtered) FINETUNING_CONFIG="admet_toymix_bbbc047" ;; # sub_module: bbbc047 (same finetuning config)
        *)              FINETUNING_CONFIG="admet" ;;
    esac
fi

# ── Cosine / 200-epoch variant ───────────────────────────────────────────────
# If USE_COSINE=1 is set, swap to the *_cosine YAML sibling of the selected
# finetuning config. The _cosine files live next to the base ones in
# expts/hydra-configs/finetuning/ and override only the scheduler + max_epochs.
COSINE_TAG=""
if [[ "${USE_COSINE:-0}" == "1" ]]; then
    FINETUNING_CONFIG="${FINETUNING_CONFIG}_cosine"
    COSINE_TAG=",'cosine_100ep'"
fi

TAGS="['${MODEL}','finetune','admet','${PRETRAIN_DATASET}'${MODEL_TAG:+,'${MODEL_TAG}'}${COSINE_TAG}]"

echo "=== Fine-tuning ${MODEL} on ADMET (ckpt=${CKPT}, pretrain=${PRETRAIN_DATASET}, unfreeze=${UNFREEZE_DEPTH}) ==="

for task in "${ADMET_TASKS[@]}"; do
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
        ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
        ++trainer.model_checkpoint.dirpath=models_checkpoints/admet/${PRETRAIN_DATASET}/${MODEL}/${task}/ \
        ${SUB_MODULE:+++finetuning.sub_module_from_pretrained=${SUB_MODULE}} \
        ${EXTRA_FLAGS:-} \
    || echo "WARN: task ${task} failed, continuing..."

    sleep 1
done
