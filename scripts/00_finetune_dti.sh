#!/usr/bin/env bash
# Fine-tune (linear-probe) a pretrained molecule encoder on TDC DTI subsets.
#
# Usage:
#   bash scripts/00_finetune_dti.sh <model> <checkpoint> [gpu_id]
#
# Arguments:
#   model      : pairmixer_12M (others need a matching training/model/dti_eval_<model>.yaml)
#   checkpoint : path to pre-trained .ckpt file (must contain a `dti_pactivity` task head)
#   gpu_id     : CUDA device index (default: $DEVICE in common.sh)
#
# Environment variables (optional):
#   DTI_SUBSETS=DAVIS,KIBA            : comma-separated TDC DTI subsets to evaluate
#   DTI_METHODS=random,cold_target    : comma-separated split methods
#   DTI_SEEDS=0,1,2,3,4               : comma-separated seeds (one run per seed)
#   DTI_EVAL_DIR=data/dti-eval        : where the prep script wrote CSVs + splits/
#   FINETUNE_DIM=256                  : hidden_dims of the new task head + finetune_head
#   ADDED_DEPTH=2                     : depth of the rebuilt task head (after drop)
#   FINETUNING_CONFIG=dti_eval        : hydra finetuning config name
#
# Examples:
#   bash scripts/00_finetune_dti.sh pairmixer_12M \
#       models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/.../last.ckpt 6
#
#   DTI_SUBSETS=DAVIS DTI_METHODS=cold_target DTI_SEEDS=0 \
#       bash scripts/00_finetune_dti.sh pairmixer_12M ./ckpt.ckpt 0

source "$(dirname "$0")/common.sh"

MODEL=${1:?  "Usage: $0 <model> <checkpoint> [gpu_id]"}
CKPT=${2:?   "Usage: $0 <model> <checkpoint> [gpu_id]"}
DEVICE=${3:-${DEVICE}}

DTI_SUBSETS=${DTI_SUBSETS:-DAVIS,KIBA}
DTI_METHODS=${DTI_METHODS:-random,cold_target}
DTI_SEEDS=${DTI_SEEDS:-0,1,2,3,4}
DTI_EVAL_DIR=${DTI_EVAL_DIR:-data/dti-eval}
FINETUNE_DIM=${FINETUNE_DIM:-256}
ADDED_DEPTH=${ADDED_DEPTH:-2}
FINETUNING_CONFIG=${FINETUNING_CONFIG:-dti_eval}

# Sanity-check the prepped data lives where the configs expect it.
if [[ ! -d "${DTI_EVAL_DIR}/splits" ]]; then
    echo "ERROR: ${DTI_EVAL_DIR}/splits/ not found."
    echo "       Run scripts/data/dti_eval/01_prepare_dti_eval.py first."
    exit 1
fi

IFS=',' read -ra SUBSET_LIST <<< "${DTI_SUBSETS}"
IFS=',' read -ra METHOD_LIST <<< "${DTI_METHODS}"
IFS=',' read -ra SEED_LIST <<< "${DTI_SEEDS}"

echo "=== DTI eval: ${MODEL} ==="
echo "    ckpt    : ${CKPT}"
echo "    subsets : ${SUBSET_LIST[*]}"
echo "    methods : ${METHOD_LIST[*]}"
echo "    seeds   : ${SEED_LIST[*]}"

for subset in "${SUBSET_LIST[@]}"; do
    for method in "${METHOD_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            csv_path="${DTI_EVAL_DIR}/${subset}.csv"
            split_path="${DTI_EVAL_DIR}/splits/${subset}_${method}_seed${seed}.pt"
            if [[ ! -f "${csv_path}" || ! -f "${split_path}" ]]; then
                echo "WARN: missing ${csv_path} or ${split_path}; skipping."
                continue
            fi

            TAGS="['${MODEL}','dti_eval','${subset}','${method}','seed${seed}']"
            echo "--- ${subset} / ${method} / seed=${seed} ---"

            SEED=${seed} CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
                model=${MODEL} \
                accelerator=gpu \
                tasks=dti_eval \
                training=dti_eval \
                $(wandb_flags "${TAGS}") \
                ++constants.dti_subset=${subset} \
                ++constants.dti_split_method=${method} \
                ++constants.dti_split_seed=${seed} \
                ++constants.dti_eval_dir=${DTI_EVAL_DIR} \
                +finetuning=${FINETUNING_CONFIG} \
                ++finetuning.pretrained_model=${CKPT} \
                ++finetuning.finetuning_head.in_dim=${FINETUNE_DIM} \
                ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
                ++finetuning.new_out_dim=${FINETUNE_DIM} \
                ++finetuning.added_depth=${ADDED_DEPTH} \
                ++finetuning.finetuning_head.depth=${ADDED_DEPTH} \
                ${EXTRA_FLAGS:-} \
            || echo "WARN: ${subset}/${method}/seed=${seed} failed, continuing..."

            sleep 1
        done
    done
done
