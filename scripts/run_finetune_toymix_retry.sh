#!/usr/bin/env bash
# Retry failed ADMET finetuning tasks for ToyMix-pretrained GPS++ 800M.
#
# Usage:
#   bash scripts/run_finetune_toymix_retry.sh [gpu_id]

set -euo pipefail
source "$(dirname "$0")/common.sh"

DEVICE=${1:-4}
FINETUNE_DIM=256
ADDED_DEPTH=4
FINETUNING_CONFIG=admet

run_task() {
    local CKPT=$1
    local task=$2
    local TAGS=$3

    echo "--- Task: ${task} (ckpt: $(basename $(dirname ${CKPT}))) ---"

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=gpspp \
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
        ++finetuning.unfreeze_pretrained_depth=0 \
        ++finetuning.epoch_unfreeze_all=none \
        ++finetuning.finetuning_head.in_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM} \
        ++finetuning.new_out_dim=${FINETUNE_DIM} \
        ++finetuning.finetuning_head.depth=${ADDED_DEPTH} \
        ++finetuning.added_depth=${ADDED_DEPTH} \
        ++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM} \
        ++trainer.model_checkpoint.save_last=False \
    || echo "WARN: ${task} failed, continuing..."

    sleep 1
}

# ── Toymix 100%: missing hia_hou, pgp_broccatelli ─────────────────────────
CKPT_100=/home/shpark/prj-molrepr/graphium/models_checkpoints/small-dataset/gpspp_800M/2026-03-17_21-26-42_20260317_212642/last.ckpt
TAGS_100="['gpspp','finetune','admet','toymix','gpspp_800M']"

echo "=== Toymix 100%: retrying 2 failed tasks (GPU ${DEVICE}) ==="
run_task "${CKPT_100}" hia_hou "${TAGS_100}"
run_task "${CKPT_100}" pgp_broccatelli "${TAGS_100}"

# ── Toymix 50%: missing caco2_wang ─────────────────────────────────────────
CKPT_50=/home/shpark/prj-molrepr/graphium/models_checkpoints/small-dataset/gpspp_800M/2026-03-18_14-53-57_20260318_145357/last.ckpt
TAGS_50="['gpspp','finetune','admet','toymix_50pct','gpspp_800M']"

echo "=== Toymix 50%: retrying 1 failed task (GPU ${DEVICE}) ==="
run_task "${CKPT_50}" caco2_wang "${TAGS_50}"

echo "=== All retries done ==="
