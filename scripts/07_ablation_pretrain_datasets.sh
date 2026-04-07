#!/usr/bin/env bash
# ============================================================================
# ABLATION: Pre-training dataset comparison for GPSPP 800M -> ADMET
# ============================================================================
# Compares how different pre-training datasets affect downstream ADMET
# performance. Uses identical architecture (GPSPP 800M) and fine-tuning
# config across all conditions. Only the pre-training data (and thus the
# prediction head used to initialize fine-tuning) differs.
#
# Conditions:
#   scratch        — no pre-training (random init baseline)
#   toymix         — molecular properties (QM9+Tox21+ZINC), init from zinc head
#   rxrx3          — cell morphology (RxRx3, 384d), init from rxrx3 head
#   toymix_rxrx3   — molecular + RxRx3, init from rxrx3 head
#   toymix_bbbc047 — molecular + BBBC047 (672d), init from bbbc047 head
#
# Usage:
#   bash scripts/07_ablation_pretrain_datasets.sh [start_gpu]
#
# Environment variables (fine-tuning config — identical for all conditions):
#   FT_UNFREEZE   : unfreeze_pretrained_depth (default: 0)
#   FT_EPOCH_ALL  : epoch_unfreeze_all (default: 50)
#   FT_DIM        : finetuning head hidden dim (default: 256)
#   FT_DEPTH      : finetuning head depth (default: 4)
#   FT_LR         : finetuning learning rate (default: 1e-4)
#   SEED          : random seed (default: 0)
#   EXTRA_FLAGS   : additional Hydra CLI flags
#
# Results are written to results/experiment_results.csv.
# Filter by wandb_tags containing 'ablation_dataset' to find these runs.
# ============================================================================

source "$(dirname "$0")/common.sh"

MODEL=gpspp_800M
START_GPU=${1:-0}

# ── Fine-tuning config (identical for all conditions) ───────────────────────
FT_UNFREEZE=${FT_UNFREEZE:-0}
FT_EPOCH_ALL=${FT_EPOCH_ALL:-50}
FT_DIM=${FT_DIM:-256}
FT_DEPTH=${FT_DEPTH:-4}
FT_LR=${FT_LR:-1e-4}

# ── 5 representative ADMET tasks (one per category) ─────────────────────────
ABLATION_TASKS=(
    lipophilicity_astrazeneca   # Absorption (regression)
    bbb_martins                 # Distribution (classification)
    cyp3a4_veith                # Metabolism (classification)
    half_life_obach             # Excretion (regression)
    ld50_zhu                    # Toxicity (regression)
)

# ── Dataset conditions ──────────────────────────────────────────────────────
# Format: "name|checkpoint_path|finetuning_config"
# "scratch" has no checkpoint or finetuning config (trained from random init)
DATASETS=(
    "scratch||"
    "toymix|models_checkpoints/small-dataset/gpspp_800M/2026-03-29_10-49-03_20260329_104903/last.ckpt|admet"
    "rxrx3|models_checkpoints/rxrx3/gpspp_800M/2026-03-18_11-25-09_20260318_112509/last.ckpt|admet_rxrx3"
    "toymix_rxrx3|models_checkpoints/toymix-rxrx3/gpspp_800M/2026-03-21_23-23-12_20260321_232312/last.ckpt|admet_toymix_rxrx3"
    "toymix_bbbc047|models_checkpoints/toymix_bbbc047/gpspp_800M/2026-03-26_17-42-01_20260326_174201/last.ckpt|admet_toymix_bbbc047"
)

NUM_CONDITIONS=${#DATASETS[@]}

echo "============================================================"
echo "  DATASET COMPARISON: GPSPP 800M -> ADMET"
echo "  Conditions: ${NUM_CONDITIONS}, Tasks: ${#ABLATION_TASKS[@]}"
echo "  Total runs: $(( NUM_CONDITIONS * ${#ABLATION_TASKS[@]} ))"
echo "  Fine-tuning: unfreeze=${FT_UNFREEZE} epoch_all=${FT_EPOCH_ALL}"
echo "               dim=${FT_DIM} depth=${FT_DEPTH} lr=${FT_LR}"
echo "  GPUs: ${START_GPU}..$(( START_GPU + NUM_CONDITIONS - 1 ))"
echo "  Seed: ${SEED}"
echo "============================================================"
echo ""

# ── Validate checkpoints ───────────────────────────────────────────────────
echo "Checkpoint validation:"
ALL_OK=true
for ds_str in "${DATASETS[@]}"; do
    IFS='|' read -r name ckpt ft_cfg <<< "${ds_str}"
    if [[ "${name}" == "scratch" ]]; then
        printf "  %-18s (no checkpoint needed)\n" "${name}"
    elif [[ -f "${ckpt}" ]]; then
        printf "  %-18s OK  %s\n" "${name}" "${ckpt}"
    else
        printf "  %-18s MISSING  %s\n" "${name}" "${ckpt}"
        ALL_OK=false
    fi
done
echo ""

if [[ "${ALL_OK}" != "true" ]]; then
    echo "Error: some checkpoints are missing. Aborting."
    exit 1
fi

# ── Worker: run fine-tuning for one condition ──────────────────────────────
run_condition() {
    local gpu=$1
    local name=$2
    local ckpt=$3
    local ft_cfg=$4

    for task in "${ABLATION_TASKS[@]}"; do
        echo "[GPU ${gpu}] ${name} / ${task}"

        TAGS="['${MODEL}','finetune','admet','${name}','ablation_dataset']"

        if [[ "${name}" == "scratch" ]]; then
            # ── Scratch: train from random init (no finetuning config) ──
            CUDA_VISIBLE_DEVICES=${gpu} graphium-train \
                model=${MODEL} \
                accelerator=gpu \
                tasks=admet \
                ++constants.seed=${SEED} \
                ++constants.wandb.entity=${WANDB_ENTITY} \
                ++constants.wandb.project=${WANDB_PROJECT} \
                "++constants.wandb.tags=${TAGS}" \
                ++constants.results_csv_dir=${RESULTS_DIR} \
                ++constants.raise_train_error=False \
                ++constants.task=${task} \
                ++datamodule.args.tdc_benchmark_names=${task} \
                ++architecture.task_heads.${task}.hidden_dims=${FT_DIM} \
                ++architecture.task_heads.${task}.depth=${FT_DEPTH} \
                ++predictor.optim_kwargs.lr=${FT_LR} \
                ++trainer.model_checkpoint.dirpath=models_checkpoints/ablation_dataset/scratch/${task}/ \
                ++trainer.model_checkpoint.save_last=False \
                ${EXTRA_FLAGS:-} \
            || echo "WARN: [GPU ${gpu}] scratch/${task} failed"
        else
            # ── Fine-tuning from pre-trained checkpoint ──
            CUDA_VISIBLE_DEVICES=${gpu} graphium-train \
                model=${MODEL} \
                accelerator=gpu \
                tasks=admet \
                ++constants.seed=${SEED} \
                ++constants.wandb.entity=${WANDB_ENTITY} \
                ++constants.wandb.project=${WANDB_PROJECT} \
                "++constants.wandb.tags=${TAGS}" \
                ++constants.results_csv_dir=${RESULTS_DIR} \
                ++constants.task=${task} \
                ++finetuning.task=${task} \
                ++datamodule.args.tdc_benchmark_names=${task} \
                +finetuning=${ft_cfg} \
                ++finetuning.pretrained_model=${ckpt} \
                ++finetuning.unfreeze_pretrained_depth=${FT_UNFREEZE} \
                ++finetuning.epoch_unfreeze_all=${FT_EPOCH_ALL} \
                ++finetuning.finetuning_head.in_dim=${FT_DIM} \
                ++finetuning.finetuning_head.hidden_dims=${FT_DIM} \
                ++finetuning.new_out_dim=${FT_DIM} \
                ++finetuning.finetuning_head.depth=${FT_DEPTH} \
                ++finetuning.added_depth=${FT_DEPTH} \
                ++architecture.task_heads.${task}.hidden_dims=${FT_DIM} \
                ++predictor.optim_kwargs.lr=${FT_LR} \
                ++trainer.model_checkpoint.dirpath=models_checkpoints/ablation_dataset/${name}/${task}/ \
                ++trainer.model_checkpoint.save_last=False \
                ${EXTRA_FLAGS:-} \
            || echo "WARN: [GPU ${gpu}] ${name}/${task} failed"
        fi

        sleep 1
    done
}

# ── Launch one worker per condition ─────────────────────────────────────────
PIDS=()
for (( i=0; i<NUM_CONDITIONS; i++ )); do
    IFS='|' read -r name ckpt ft_cfg <<< "${DATASETS[$i]}"
    gpu_id=$(( START_GPU + i ))

    run_condition "${gpu_id}" "${name}" "${ckpt}" "${ft_cfg}" &
    PIDS+=($!)
    echo "Launched ${name} on GPU ${gpu_id} (PID ${PIDS[-1]})"
done

# ── Wait for all workers ────────────────────────────────────────────────────
echo ""
echo "Waiting for ${#PIDS[@]} workers (${NUM_CONDITIONS} conditions x ${#ABLATION_TASKS[@]} tasks)..."
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || (( FAILED++ ))
done

echo ""
echo "============================================================"
echo "  DATASET COMPARISON COMPLETE"
echo "  Failed workers: ${FAILED}/${#PIDS[@]}"
echo "  Results: ${RESULTS_DIR}/experiment_results.csv"
echo "  Filter by wandb_tags containing 'ablation_dataset'"
echo "============================================================"
