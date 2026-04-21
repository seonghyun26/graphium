#!/usr/bin/env bash
# Queue representative-task ADMET finetunes for the current PairMixer 12M
# ToyMix pretrain. The script waits for the active pretrain to finish, resolves
# the newest compatible last.ckpt (preferring pairmixer_12M, then pairmixer_auto),
# then runs the requested LR/scheduler comparisons on GPU 7.

set -euo pipefail

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE:-7}}
MODEL=${MODEL:-pairmixer_12M}
GRAPHIUM_TRAIN=${GRAPHIUM_TRAIN:-/home/shpark/.conda/envs/graphium/bin/graphium-train}
PRETRAIN_DATASET=${PRETRAIN_DATASET:-toymix}
WAIT_PID=${WAIT_PID:-}
WAIT_PATTERN=${WAIT_PATTERN:-graphium-train model=pairmixer_auto accelerator=gpu tasks=toymix training=toymix architecture=toymix}
POLL_SECONDS=${POLL_SECONDS:-60}
FINETUNE_DIM=${FINETUNE_DIM:-256}
ADDED_DEPTH=${ADDED_DEPTH:-4}
UNFREEZE_DEPTH=${UNFREEZE_DEPTH:-0}
EPOCH_UNFREEZE_ALL=${EPOCH_UNFREEZE_ALL:-none}
DRY_RUN=${DRY_RUN:-0}
RESULT_ROOT=${RESULT_ROOT:-models_checkpoints/admet-representative}
CKPT_OVERRIDE=${CKPT_OVERRIDE:-}

REP_TASKS=(
    caco2_wang
    bbb_martins
    cyp2d6_veith
    clearance_microsome_az
    ld50_zhu
)

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*"
}

wait_for_pretrain() {
    if [[ -n "${WAIT_PID}" ]]; then
        while kill -0 "${WAIT_PID}" 2>/dev/null; do
            log "Waiting for PID ${WAIT_PID} (current PairMixer auto ToyMix pretrain) to finish..."
            sleep "${POLL_SECONDS}"
        done
        log "PID ${WAIT_PID} has finished."
        return
    fi

    while true; do
        local matches
        matches=$(pgrep -af "${WAIT_PATTERN}" || true)
        matches=$(printf "%s\n" "${matches}" | grep -v -F "$0" || true)
        if [[ -z "${matches}" ]]; then
            log "No active pretrain matching WAIT_PATTERN remains."
            return
        fi

        log "Waiting for matching pretrain process(es) to finish:"
        printf "%s\n" "${matches}"
        sleep "${POLL_SECONDS}"
    done
}

resolve_latest_ckpt() {
    local ckpt

    if [[ "${MODEL}" == "pairmixer_12M" || "${MODEL}" == "pairmixer_auto" ]]; then
        ckpt=$(
            find models_checkpoints/small-dataset \
                \( -path "*/pairmixer_auto/*" -o -path "*/pairmixer_12M/*" \) \
                -type f -name "last.ckpt" -printf "%T@ %p\n" 2>/dev/null \
                | sort -nr \
                | head -n 1 \
                | cut -d' ' -f2-
        )
    else
        ckpt=$(
            find "models_checkpoints/small-dataset/${MODEL}" -maxdepth 2 -type f -name "last.ckpt" -printf "%T@ %p\n" 2>/dev/null \
                | sort -nr \
                | head -n 1 \
                | cut -d' ' -f2-
        )
    fi

    if [[ -z "${ckpt}" ]]; then
        log "ERROR: Could not find a PairMixer auto/12M last.ckpt under models_checkpoints/small-dataset."
        exit 1
    fi

    printf "%s\n" "${ckpt}"
}

run_task() {
    local label=$1
    local finetuning_config=$2
    local lr=$3
    local task=$4
    local tags="['${MODEL}','finetune','admet','${PRETRAIN_DATASET}','rep5','${label}']"

    log "Starting ${label} on ${task} (lr=${lr}, finetuning=${finetuning_config})"

    local cmd=(
        "${GRAPHIUM_TRAIN}"
        "model=${MODEL}"
        "accelerator=gpu"
        "tasks=admet"
        "++constants.task=${task}"
        "+finetuning=${finetuning_config}"
        "++finetuning.task=${task}"
        "++datamodule.args.tdc_benchmark_names=${task}"
        "++finetuning.pretrained_model=${CKPT}"
        "++finetuning.sub_module_from_pretrained=zinc"
        "++finetuning.unfreeze_pretrained_depth=${UNFREEZE_DEPTH}"
        "++finetuning.epoch_unfreeze_all=${EPOCH_UNFREEZE_ALL}"
        "++finetuning.finetuning_head.in_dim=${FINETUNE_DIM}"
        "++finetuning.finetuning_head.hidden_dims=${FINETUNE_DIM}"
        "++finetuning.new_out_dim=${FINETUNE_DIM}"
        "++finetuning.finetuning_head.depth=${ADDED_DEPTH}"
        "++finetuning.added_depth=${ADDED_DEPTH}"
        "++architecture.task_heads.${task}.hidden_dims=${FINETUNE_DIM}"
        "++predictor.optim_kwargs.lr=${lr}"
        "++constants.name=finetuning_${task}_${MODEL}_${label}"
        "++trainer.model_checkpoint.dirpath=${RESULT_ROOT}/${PRETRAIN_DATASET}/${MODEL}/${label}/${task}/"
    )

    local wandb_cmd
    wandb_cmd=$(wandb_flags "${tags}")

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf "CUDA_VISIBLE_DEVICES=%s " "${DEVICE}"
        printf "%q " "${cmd[@]}"
        printf "%s" "${wandb_cmd}"
        printf "\n"
        return
    fi

    CUDA_VISIBLE_DEVICES=${DEVICE} "${cmd[@]}" ${wandb_cmd}
}

run_variant() {
    local label=$1
    local finetuning_config=$2
    local lr=$3

    for task in "${REP_TASKS[@]}"; do
        run_task "${label}" "${finetuning_config}" "${lr}" "${task}"
        sleep 1
    done
}

main() {
    log "Representative ADMET tasks: ${REP_TASKS[*]}"
    log "Target GPU: ${DEVICE}"
    wait_for_pretrain

    if [[ -n "${CKPT_OVERRIDE}" ]]; then
        CKPT="${CKPT_OVERRIDE}"
    else
        CKPT=$(resolve_latest_ckpt)
    fi

    if [[ ! -f "${CKPT}" ]]; then
        log "ERROR: Checkpoint not found: ${CKPT}"
        exit 1
    fi

    export CKPT
    log "Resolved checkpoint: ${CKPT}"

    run_variant baseline_lr1e4 admet 1e-4
    run_variant warmup_cosine_lr1e4 admet_cosine 1e-4
    run_variant baseline_lr5e5 admet 5e-5
    run_variant baseline_lr2e4 admet 2e-4

    log "Queue completed."
}

main "$@"
