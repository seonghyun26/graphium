#!/usr/bin/env bash
# ============================================================================
# ABLATION: Fine-tuning config search for GPSPP 800M (toymix_bbbc047)
# ============================================================================
# Tests various fine-tuning hyperparameters using an existing pre-trained
# checkpoint. Runs 5 representative ADMET tasks per config.
#
# Usage:
#   bash scripts/05_ablation_bbbc047_finetune.sh [checkpoint] [start_gpu] [num_gpus]
#
# Arguments:
#   checkpoint : path to pre-trained .ckpt (default: auto-detect latest)
#   start_gpu  : first GPU index (default: 0)
#   num_gpus   : number of GPUs to use (default: 8)
#
# Environment variables:
#   SKIP_FROZEN  : set to 1 to skip the frozen baseline (already run)
#   SEED         : random seed (default: 0)
#   EXTRA_FLAGS  : additional Hydra CLI flags
#
# Results are written to results/experiment_results.csv.
# Filter by wandb_tags containing 'ablation_ft' to find these runs.
# ============================================================================

source "$(dirname "$0")/common.sh"

MODEL=gpspp_800M
CKPT=${1:-models_checkpoints/toymix_bbbc047/gpspp_800M/2026-03-26_17-42-01_20260326_174201/last.ckpt}
START_GPU=${2:-0}
NUM_GPUS=${3:-8}
SKIP_FROZEN=${SKIP_FROZEN:-1}

if [[ ! -f "${CKPT}" ]]; then
    echo "Error: checkpoint not found: ${CKPT}"
    exit 1
fi

# ── 5 representative ADMET tasks (one per category) ─────────────────────────
ABLATION_TASKS=(
    lipophilicity_astrazeneca   # Absorption (regression)
    bbb_martins                 # Distribution (classification)
    cyp3a4_veith                # Metabolism (classification)
    half_life_obach             # Excretion (regression)
    ld50_zhu                    # Toxicity (regression)
)

FINETUNING_CONFIG=admet_toymix_bbbc047

# ── Fine-tuning configurations to search ────────────────────────────────────
# Format: "name|unfreeze_depth|epoch_unfreeze_all|finetune_dim|added_depth|lr"
CONFIGS=(
    # Tier 1: Unfreezing strategy (default head: dim=256, depth=4, lr=1e-4)
    "frozen|0|none|256|4|1e-4"                  # baseline (fully frozen)
    "unfreeze_e50|0|50|256|4|1e-4"              # unfreeze all at epoch 50
    "unfreeze_e25|0|25|256|4|1e-4"              # unfreeze all at epoch 25
    "prog2_e50|2|50|256|4|1e-4"                 # progressive: 2 layers first, all at 50
    "prog4_e50|4|50|256|4|1e-4"                 # progressive: 4 layers first, all at 50
    "prog4_e25|4|25|256|4|1e-4"                 # aggressive progressive unfreezing

    # Tier 2: Learning rate (with unfreeze_all at epoch 50)
    "lr5e5|0|50|256|4|5e-5"                     # lower LR
    "lr2e4|0|50|256|4|2e-4"                     # higher LR

    # Tier 3: Head architecture (with unfreeze_all at epoch 50)
    "head_s|0|50|128|2|1e-4"                    # smaller head
    "head_l|0|50|512|4|1e-4"                    # larger head

    # Tier 4: Combined
    "combo1|4|25|512|4|5e-5"                    # progressive + low LR + large head
    "combo2|0|25|512|4|5e-5"                    # early unfreeze + low LR + large head
)

# ── Optionally skip frozen baseline ────────────────────────────────────────
if [[ "${SKIP_FROZEN}" == "1" ]]; then
    FILTERED_CONFIGS=()
    for cfg_str in "${CONFIGS[@]}"; do
        IFS='|' read -r name _ <<< "${cfg_str}"
        [[ "${name}" == "frozen" ]] && continue
        FILTERED_CONFIGS+=("${cfg_str}")
    done
    CONFIGS=("${FILTERED_CONFIGS[@]}")
fi

# ── Build job list ──────────────────────────────────────────────────────────
JOBS=()
for cfg_str in "${CONFIGS[@]}"; do
    IFS='|' read -r name unfreeze epoch_all dim depth lr <<< "${cfg_str}"
    for task in "${ABLATION_TASKS[@]}"; do
        JOBS+=("${name}|${unfreeze}|${epoch_all}|${dim}|${depth}|${lr}|${task}")
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo "============================================================"
echo "  FINE-TUNING CONFIG SEARCH: GPSPP 800M (toymix_bbbc047)"
echo "  Checkpoint: ${CKPT}"
echo "  Configs: ${#CONFIGS[@]}, Tasks: ${#ABLATION_TASKS[@]}, Total jobs: ${TOTAL_JOBS}"
echo "  GPUs: ${START_GPU}..$(( START_GPU + NUM_GPUS - 1 ))"
echo "  Seed: ${SEED}"
echo "============================================================"
echo ""
echo "Configs to test:"
for cfg_str in "${CONFIGS[@]}"; do
    IFS='|' read -r name unfreeze epoch_all dim depth lr <<< "${cfg_str}"
    printf "  %-16s unfreeze=%s epoch_all=%-4s dim=%-3s depth=%s lr=%s\n" \
        "${name}" "${unfreeze}" "${epoch_all}" "${dim}" "${depth}" "${lr}"
done
echo ""

# ── Distribute jobs across GPUs ─────────────────────────────────────────────
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

for (( g=0; g<NUM_GPUS; g++ )); do
    : > "${TMPDIR}/gpu_${g}.sh"
done

for (( i=0; i<TOTAL_JOBS; i++ )); do
    gpu_idx=$(( i % NUM_GPUS ))
    echo "${JOBS[$i]}" >> "${TMPDIR}/gpu_${gpu_idx}.sh"
done

# Print GPU assignment summary
for (( g=0; g<NUM_GPUS; g++ )); do
    gpu_id=$(( START_GPU + g ))
    n_jobs=$(wc -l < "${TMPDIR}/gpu_${g}.sh")
    echo "  GPU ${gpu_id}: ${n_jobs} jobs"
done
echo ""

# ── Worker function ─────────────────────────────────────────────────────────
run_gpu_jobs() {
    local gpu=$1
    local jobfile=$2

    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        IFS='|' read -r name unfreeze epoch_all dim depth lr task <<< "${line}"

        echo "[GPU ${gpu}] config=${name} task=${task}"

        TAGS="['${MODEL}','finetune','admet','toymix_bbbc047','ablation_ft','${name}']"

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
            +finetuning=${FINETUNING_CONFIG} \
            ++finetuning.pretrained_model=${CKPT} \
            ++finetuning.unfreeze_pretrained_depth=${unfreeze} \
            ++finetuning.epoch_unfreeze_all=${epoch_all} \
            ++finetuning.finetuning_head.in_dim=${dim} \
            ++finetuning.finetuning_head.hidden_dims=${dim} \
            ++finetuning.new_out_dim=${dim} \
            ++finetuning.finetuning_head.depth=${depth} \
            ++finetuning.added_depth=${depth} \
            ++architecture.task_heads.${task}.hidden_dims=${dim} \
            ++predictor.optim_kwargs.lr=${lr} \
            ++trainer.model_checkpoint.dirpath=models_checkpoints/ablation_ft/toymix_bbbc047/${MODEL}/${name}/${task}/ \
            ++trainer.model_checkpoint.save_last=False \
            ${EXTRA_FLAGS:-} \
        || echo "WARN: [GPU ${gpu}] config=${name} task=${task} failed"

        sleep 1
    done < "${jobfile}"
}

# ── Launch parallel workers ─────────────────────────────────────────────────
PIDS=()
for (( g=0; g<NUM_GPUS; g++ )); do
    gpu_id=$(( START_GPU + g ))
    jobfile="${TMPDIR}/gpu_${g}.sh"

    if [[ -s "${jobfile}" ]]; then
        run_gpu_jobs "${gpu_id}" "${jobfile}" &
        PIDS+=($!)
        echo "Launched GPU ${gpu_id} worker (PID ${PIDS[-1]})"
    fi
done

# ── Wait for all workers ────────────────────────────────────────────────────
echo ""
echo "Waiting for ${#PIDS[@]} GPU workers to complete..."
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || (( FAILED++ ))
done

echo ""
echo "============================================================"
echo "  FINE-TUNING CONFIG SEARCH COMPLETE"
echo "  Failed workers: ${FAILED}/${#PIDS[@]}"
echo "  Results: ${RESULTS_DIR}/experiment_results.csv"
echo "  Filter by wandb_tags containing 'ablation_ft'"
echo "============================================================"
