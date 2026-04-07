#!/usr/bin/env bash
# ============================================================================
# ABLATION: Pre-training config search for GPSPP 800M (toymix_bbbc047)
# ============================================================================
# Tests various pre-training hyperparameters, then fine-tunes each checkpoint
# on 5 representative ADMET tasks. Max epochs are NOT changed (100 for both).
#
# Usage:
#   bash scripts/06_ablation_bbbc047_pretrain.sh [start_gpu]
#
# Arguments:
#   start_gpu : first GPU index (default: 0)
#
# Environment variables (fine-tuning config — set to best from finetune search):
#   FT_UNFREEZE   : unfreeze_pretrained_depth (default: 0)
#   FT_EPOCH_ALL  : epoch_unfreeze_all (default: 50)
#   FT_DIM        : finetuning head hidden dim (default: 256)
#   FT_DEPTH      : finetuning head depth (default: 4)
#   FT_LR         : finetuning learning rate (default: 1e-4)
#   FT_CONFIG_NAME: label for wandb tags (default: default_ft)
#
# Control:
#   SKIP_PRETRAIN  : set to 1 to skip pre-training (use existing checkpoints)
#   SEED           : random seed (default: 0)
#   EXTRA_FLAGS    : additional Hydra CLI flags
#
# Results are written to results/experiment_results.csv.
# Filter by wandb_tags containing 'ablation_pt' to find these runs.
# ============================================================================

source "$(dirname "$0")/common.sh"

MODEL=gpspp_800M
DATASET=toymix_bbbc047
START_GPU=${1:-0}

# ── Fine-tuning config (override with best from finetune search) ────────────
FT_UNFREEZE=${FT_UNFREEZE:-0}
FT_EPOCH_ALL=${FT_EPOCH_ALL:-50}
FT_DIM=${FT_DIM:-256}
FT_DEPTH=${FT_DEPTH:-4}
FT_LR=${FT_LR:-1e-4}
FT_CONFIG_NAME=${FT_CONFIG_NAME:-default_ft}

# ── 5 representative ADMET tasks (one per category) ─────────────────────────
ABLATION_TASKS=(
    lipophilicity_astrazeneca   # Absorption (regression)
    bbb_martins                 # Distribution (classification)
    cyp3a4_veith                # Metabolism (classification)
    half_life_obach             # Excretion (regression)
    ld50_zhu                    # Toxicity (regression)
)

FINETUNING_CONFIG=admet_toymix_bbbc047
CKPT_BASE=models_checkpoints/ablation_pt/toymix_bbbc047/${MODEL}

# ── Pre-training configurations ─────────────────────────────────────────────
# Format: "name|lr|warmup_epochs|weight_decay"
# Batch size stays at 100 (from training/model config)
PT_CONFIGS=(
    "pt_lr1e4|1e-4|5|0"                    # higher LR, shorter warmup
    "pt_wd|4e-5|10|1e-7"                   # with weight decay
    "pt_lr1e4_wd|1e-4|5|1e-7"             # higher LR + weight decay
)

NUM_PT=${#PT_CONFIGS[@]}

echo "============================================================"
echo "  PRE-TRAINING CONFIG SEARCH: GPSPP 800M (toymix_bbbc047)"
echo "  Pre-training configs: ${NUM_PT} (+ existing baseline)"
echo "  Fine-tuning config: ${FT_CONFIG_NAME}"
echo "    unfreeze=${FT_UNFREEZE} epoch_all=${FT_EPOCH_ALL}"
echo "    dim=${FT_DIM} depth=${FT_DEPTH} lr=${FT_LR}"
echo "  GPUs: ${START_GPU}..$(( START_GPU + NUM_PT - 1 )) (pre-training)"
echo "  Seed: ${SEED}"
echo "============================================================"
echo ""
echo "Pre-training configs:"
for cfg_str in "${PT_CONFIGS[@]}"; do
    IFS='|' read -r name lr warmup wd <<< "${cfg_str}"
    printf "  %-16s lr=%s warmup=%s wd=%s\n" "${name}" "${lr}" "${warmup}" "${wd}"
done
echo ""

# ── Phase 1: Pre-training ───────────────────────────────────────────────────
if [[ "${SKIP_PRETRAIN:-0}" != "1" ]]; then
    echo "=== Phase 1: Pre-training (${NUM_PT} configs in parallel) ==="

    PT_PIDS=()
    for (( i=0; i<NUM_PT; i++ )); do
        IFS='|' read -r name lr warmup wd <<< "${PT_CONFIGS[$i]}"
        gpu_id=$(( START_GPU + i ))
        ckpt_dir="${CKPT_BASE}/${name}"

        echo "  Launching ${name} on GPU ${gpu_id} (lr=${lr}, warmup=${warmup}, wd=${wd})"

        TAGS="['${MODEL}','pretrain','${DATASET}','ablation_pt','${name}']"

        mkdir -p "${ckpt_dir}"

        # Build weight decay flag conditionally
        WD_FLAG=""
        if [[ "${wd}" != "0" ]]; then
            WD_FLAG="++predictor.optim_kwargs.weight_decay=${wd}"
        fi

        CUDA_VISIBLE_DEVICES=${gpu_id} graphium-train \
            model=${MODEL} \
            accelerator=gpu \
            tasks=toymix_bbbc047 \
            training=toymix_bbbc047 \
            architecture=toymix \
            ++constants.seed=${SEED} \
            ++constants.wandb.entity=${WANDB_ENTITY} \
            ++constants.wandb.project=${WANDB_PROJECT} \
            "++constants.wandb.tags=${TAGS}" \
            ++constants.results_csv_dir=${RESULTS_DIR} \
            ++predictor.optim_kwargs.lr=${lr} \
            ++predictor.torch_scheduler_kwargs.warmup_epochs=${warmup} \
            ${WD_FLAG} \
            ++trainer.model_checkpoint.dirpath=${ckpt_dir} \
            ${EXTRA_FLAGS:-} \
        &
        PT_PIDS+=($!)
    done

    echo ""
    echo "Waiting for ${#PT_PIDS[@]} pre-training runs..."
    PT_FAILED=0
    for pid in "${PT_PIDS[@]}"; do
        wait "${pid}" || (( PT_FAILED++ ))
    done
    echo "Pre-training done. Failed: ${PT_FAILED}/${#PT_PIDS[@]}"
    echo ""
else
    echo "=== Phase 1: SKIPPED (SKIP_PRETRAIN=1) ==="
    echo ""
fi

# ── Phase 2: Fine-tuning all checkpoints ───────────────────────────────────
echo "=== Phase 2: Fine-tuning all checkpoints on ADMET ==="

# Collect checkpoint paths (baseline + new)
declare -A CKPT_MAP
CKPT_MAP["pt_baseline"]="models_checkpoints/toymix_bbbc047/gpspp_800M/2026-03-26_17-42-01_20260326_174201/last.ckpt"

for cfg_str in "${PT_CONFIGS[@]}"; do
    IFS='|' read -r name _ _ _ <<< "${cfg_str}"
    ckpt_path="${CKPT_BASE}/${name}/last.ckpt"
    if [[ -f "${ckpt_path}" ]]; then
        CKPT_MAP["${name}"]="${ckpt_path}"
    else
        echo "WARN: checkpoint not found: ${ckpt_path}, skipping ${name}"
    fi
done

echo "Available checkpoints:"
for pt_name in "${!CKPT_MAP[@]}"; do
    echo "  ${pt_name}: ${CKPT_MAP[$pt_name]}"
done
echo ""

# Build fine-tuning jobs
FT_JOBS=()
for pt_name in $(echo "${!CKPT_MAP[@]}" | tr ' ' '\n' | sort); do
    ckpt="${CKPT_MAP[$pt_name]}"
    for task in "${ABLATION_TASKS[@]}"; do
        FT_JOBS+=("${pt_name}|${ckpt}|${task}")
    done
done

TOTAL_FT=${#FT_JOBS[@]}
# Use all available GPUs for fine-tuning (up to NUM_PT + 1)
FT_NUM_GPUS=$(( NUM_PT + 1 ))
(( FT_NUM_GPUS > 8 )) && FT_NUM_GPUS=8

echo "Fine-tuning: ${TOTAL_FT} jobs across ${FT_NUM_GPUS} GPUs"

# Distribute across GPUs
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

for (( g=0; g<FT_NUM_GPUS; g++ )); do
    : > "${TMPDIR}/gpu_${g}.sh"
done
for (( i=0; i<TOTAL_FT; i++ )); do
    echo "${FT_JOBS[$i]}" >> "${TMPDIR}/gpu_$(( i % FT_NUM_GPUS )).sh"
done

# Worker function
run_ft_gpu() {
    local gpu=$1
    local jobfile=$2

    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        IFS='|' read -r pt_name ckpt task <<< "${line}"

        echo "[GPU ${gpu}] ${pt_name} / ${task}"

        TAGS="['${MODEL}','finetune','admet','toymix_bbbc047','ablation_pt','${pt_name}','${FT_CONFIG_NAME}']"

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
            ++trainer.model_checkpoint.dirpath=models_checkpoints/ablation_pt/finetune/${pt_name}/${task}/ \
            ++trainer.model_checkpoint.save_last=False \
            ${EXTRA_FLAGS:-} \
        || echo "WARN: [GPU ${gpu}] ${pt_name}/${task} failed"

        sleep 1
    done < "${jobfile}"
}

# Launch workers
FT_PIDS=()
for (( g=0; g<FT_NUM_GPUS; g++ )); do
    gpu_id=$(( START_GPU + g ))
    jobfile="${TMPDIR}/gpu_${g}.sh"
    if [[ -s "${jobfile}" ]]; then
        run_ft_gpu "${gpu_id}" "${jobfile}" &
        FT_PIDS+=($!)
        echo "Launched GPU ${gpu_id} fine-tuning worker (PID ${FT_PIDS[-1]})"
    fi
done

echo ""
echo "Waiting for ${#FT_PIDS[@]} fine-tuning workers..."
FT_FAILED=0
for pid in "${FT_PIDS[@]}"; do
    wait "${pid}" || (( FT_FAILED++ ))
done

echo ""
echo "============================================================"
echo "  PRE-TRAINING CONFIG SEARCH COMPLETE"
echo "  Pre-training failures: ${PT_FAILED:-0}/${NUM_PT}"
echo "  Fine-tuning failures: ${FT_FAILED}/${#FT_PIDS[@]}"
echo "  Results: ${RESULTS_DIR}/experiment_results.csv"
echo "  Filter by wandb_tags containing 'ablation_pt'"
echo "============================================================"
