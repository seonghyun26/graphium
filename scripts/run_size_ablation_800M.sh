#!/usr/bin/env bash
# ============================================================================
# Dataset Size Ablation: GPS++ 800M
# ============================================================================
# Pre-trains on subsampled fractions of a dataset, then fine-tunes on ADMET.
# Uses constants.sample_size (wired into all task_specific_args via Hydra).
#
# Usage:
#   bash scripts/run_size_ablation_800M.sh <dataset> <gpu_id>
#
# Arguments:
#   dataset : largemix | toymix | rxrx3
#   gpu_id  : CUDA device index
#
# Examples:
#   bash scripts/run_size_ablation_800M.sh largemix 5
#   FRACTIONS="0.1 0.5 1.0" bash scripts/run_size_ablation_800M.sh toymix 4

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${1:? "Usage: $0 <dataset> <gpu_id>"}
GPU=${2:? "Usage: $0 <dataset> <gpu_id>"}
SEED=${SEED:-0}
FRACTIONS=${FRACTIONS:-"0.01 0.05 0.1 0.25 0.5 1.0"}

# Dataset -> Hydra overrides
case "${DATASET}" in
    toymix)       TRAINING=toymix;   ARCHITECTURE=toymix;   TASKS=toymix ;;
    largemix)     TRAINING=largemix; ARCHITECTURE=largemix;  TASKS=largemix ;;
    rxrx3)        TRAINING=rxrx3;    ARCHITECTURE=largemix;  TASKS=rxrx3 ;;
    largemix_rxrx3) TRAINING=largemix; ARCHITECTURE=largemix; TASKS=largemix_rxrx3 ;;
    *) echo "Error: unknown dataset '${DATASET}'"; exit 1 ;;
esac

# Build per-task sample_size overrides
build_sample_flags() {
    local frac=$1
    local flags=""
    case "${DATASET}" in
        toymix)
            for t in qm9 tox21 zinc; do
                flags="${flags} ++datamodule.args.task_specific_args.${t}.sample_size=${frac}"
            done ;;
        largemix|largemix_rxrx3)
            for t in l1000_vcap l1000_mcf7 pcba_1328 pcqm4m_g25 pcqm4m_n4; do
                flags="${flags} ++datamodule.args.task_specific_args.${t}.sample_size=${frac}"
            done ;;
        rxrx3)
            flags="++datamodule.args.task_specific_args.rxrx3.sample_size=${frac}" ;;
    esac
    if [[ "${DATASET}" == "largemix_rxrx3" ]]; then
        flags="${flags} ++datamodule.args.task_specific_args.rxrx3.sample_size=${frac}"
    fi
    echo "${flags}"
}

CKPT_BASE="/home/shpark/prj-molrepr/graphium/models_checkpoints/size_ablation"

echo "============================================================"
echo "  SIZE ABLATION: GPS++ 800M on ${DATASET}"
echo "  Fractions: ${FRACTIONS}"
echo "  GPU: ${GPU}"
echo "============================================================"

for FRAC in ${FRACTIONS}; do
    CKPT_DIR="${CKPT_BASE}/${DATASET}_frac${FRAC}"
    mkdir -p "${CKPT_DIR}"

    echo ""
    echo ">>> Pre-training on ${DATASET} with sample_size=${FRAC} <<<"

    CUDA_VISIBLE_DEVICES=${GPU} graphium-train \
        model=gpspp_800M \
        accelerator=gpu \
        tasks=${TASKS} \
        training=${TRAINING} \
        architecture=${ARCHITECTURE} \
        ++constants.seed=${SEED} \
        $(build_sample_flags ${FRAC}) \
        ++constants.wandb.entity=eddy26 \
        ++constants.wandb.save_dir=null \
        ++constants.wandb.project=graphium \
        "++constants.wandb.tags=['gpspp','gpspp_800M','pretrain','${DATASET}','frac_${FRAC}','size_ablation']" \
        ++constants.wandb.name=pretrain_800M_${DATASET}_frac${FRAC} \
        ++trainer.model_checkpoint.dirpath=${CKPT_DIR} \
        ++trainer.model_checkpoint.save_last=True \
    || { echo "WARN: pre-training frac=${FRAC} failed, skipping finetune"; continue; }

    echo ""
    echo ">>> Fine-tuning frac=${FRAC} checkpoint on ADMET <<<"

    CKPT_FILE="${CKPT_DIR}/last.ckpt"
    if [[ ! -f "${CKPT_FILE}" ]]; then
        echo "WARN: checkpoint not found at ${CKPT_FILE}, skipping"
        continue
    fi

    source "$(dirname "$0")/common.sh"
    for task in "${ADMET_TASKS[@]}"; do
        echo "--- frac=${FRAC} task=${task} ---"

        CUDA_VISIBLE_DEVICES=${GPU} graphium-train \
            model=gpspp_800M \
            accelerator=gpu \
            tasks=admet \
            ++constants.seed=${SEED} \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            "++constants.wandb.tags=['gpspp','gpspp_800M','finetune','admet','${DATASET}','frac_${FRAC}','size_ablation']" \
            ++constants.raise_train_error=False \
            ++constants.task=${task} \
            ++finetuning.task=${task} \
            ++datamodule.args.tdc_benchmark_names=${task} \
            ++datamodule.args.num_workers=0 \
            ++datamodule.args.featurization_n_jobs=0 \
            +finetuning=admet \
            ++finetuning.pretrained_model=${CKPT_FILE} \
            ++finetuning.unfreeze_pretrained_depth=0 \
            ++finetuning.epoch_unfreeze_all=none \
            ++finetuning.finetuning_head.in_dim=256 \
            ++finetuning.finetuning_head.hidden_dims=256 \
            ++finetuning.new_out_dim=256 \
            ++finetuning.finetuning_head.depth=4 \
            ++finetuning.added_depth=4 \
            ++architecture.task_heads.${task}.hidden_dims=256 \
            ++trainer.model_checkpoint.save_last=False \
        || echo "WARN: task ${task} failed, continuing..."

        sleep 1
    done
done

echo ""
echo "============================================================"
echo "  SIZE ABLATION COMPLETE"
echo "  Filter W&B by tags: size_ablation, gpspp_800M"
echo "============================================================"
