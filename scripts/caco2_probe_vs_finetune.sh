#!/usr/bin/env bash
# Compare linear probing vs fine-tuning from toymix_dti_esmc_v2 pairmixer_20M
# on 5 representative ADMET tasks (one per category).

set -euo pipefail
cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate graphium 2>/dev/null || true
export PATH="/home/shpark/.conda/envs/graphium/bin:$PATH"

source "$(dirname "$0")/common.sh"

MODEL=pairmixer_20M
CKPT="models_checkpoints/toymix-dti-esmc-v2/pairmixer_20M/2026-04-11_00-37-44_20260411_003745/last.ckpt"
DEVICE=0
FINETUNING_CONFIG=admet_toymix_dti
PRETRAIN_DATASET=toymix_dti_esmc_v2

# 5 representative tasks (one per ADMET category)
TASKS=(
    caco2_wang            # Absorption (regression, MAE)
    pgp_broccatelli       # Distribution (classification, AUROC)
    cyp3a4_veith          # Metabolism (classification, AUPRC)
    half_life_obach       # Excretion (regression, Spearman)
    herg                  # Toxicity (classification, AUROC)
)

LOG_DIR="logs/probe_vs_ft_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

run_experiment() {
    local name="$1"
    local task="$2"
    shift 2
    local extra_flags=("$@")

    echo ""
    echo "=========================================="
    echo "  ${name} | ${task} | $(date)"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=admet \
        $(wandb_flags "['${MODEL}','finetune','admet','${PRETRAIN_DATASET}','probe_vs_ft','${name}']") \
        ++constants.task=${task} \
        ++finetuning.task=${task} \
        ++datamodule.args.tdc_benchmark_names=${task} \
        ++datamodule.args.num_workers=0 \
        ++datamodule.args.featurization_n_jobs=0 \
        +finetuning=${FINETUNING_CONFIG} \
        ++finetuning.pretrained_model=${CKPT} \
        ++trainer.model_checkpoint.dirpath=models_checkpoints/probe_vs_ft/${name}/${task}/ \
        "${extra_flags[@]}" \
        2>&1 | tee "${LOG_DIR}/${name}_${task}.log"

    echo ">>> ${name} ${task} done at $(date)" >> "${LOG_DIR}/summary.txt"
    sleep 2
}

echo "Starting probe_vs_ft at $(date)" | tee "${LOG_DIR}/summary.txt"

# ============================================================================
# LINEAR PROBING: frozen backbone, only task head trains
# ============================================================================
for task in "${TASKS[@]}"; do
    run_experiment "linear_probe" "${task}" \
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=none
done

# ============================================================================
# FINE-TUNING: gradual unfreezing (best config from GPS++ experiments)
# ============================================================================
for task in "${TASKS[@]}"; do
    run_experiment "finetune" "${task}" \
        ++finetuning.unfreeze_pretrained_depth=4 \
        ++finetuning.epoch_unfreeze_all=50
done

echo ""
echo "============================================="
echo "  DONE at $(date)"
echo "============================================="
