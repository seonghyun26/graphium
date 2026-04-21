#!/usr/bin/env bash
# ============================================================================
# ABLATION: Virtual Node in PairMixer 10M (toymix pre-training)
# ============================================================================
# Compares PairMixer 10M with and without virtual node aggregation.
# Both variants are pre-trained on ToyMix, then fine-tuned on 5 ADMET tasks.
#
# Conditions:
#   pairmixer_10M     — baseline (no virtual node)
#   pairmixer_10M_vn  — with virtual node (sum aggregation)
#
# Usage:
#   bash scripts/08_ablation_pairmixer_vn.sh [gpu_baseline] [gpu_vn]
#
# Arguments:
#   gpu_baseline : GPU for baseline (default: 0)
#   gpu_vn       : GPU for VN variant (default: 1)
#
# Environment variables:
#   SEEDS          : space-separated seeds (default: "0 1 2")
#   SKIP_PRETRAIN  : set to 1 to skip pre-training (use existing checkpoints)
#   EXTRA_FLAGS    : additional Hydra CLI flags
#
# Results are written to results/experiment_results.csv.
# Filter by wandb_tags containing 'ablation_vn' to find these runs.
# ============================================================================

source "$(dirname "$0")/common.sh"

GPU_BASE=${1:-0}
GPU_VN=${2:-1}
SEEDS=(${SEEDS:-0 1 2})

MODEL_BASE=pairmixer_10M
MODEL_VN=pairmixer_10M_vn
DATASET=toymix

# ── Fine-tuning config (identical for both conditions) ──────────────────────
FT_UNFREEZE=${FT_UNFREEZE:-0}
FT_EPOCH_ALL=${FT_EPOCH_ALL:-50}
FT_DIM=${FT_DIM:-256}
FT_DEPTH=${FT_DEPTH:-4}
FT_LR=${FT_LR:-1e-4}
FINETUNING_CONFIG=admet

# ── 5 representative ADMET tasks ────────────────────────────────────────────
ABLATION_TASKS=(
    lipophilicity_astrazeneca   # Absorption (regression)
    bbb_martins                 # Distribution (classification)
    cyp3a4_veith                # Metabolism (classification)
    half_life_obach             # Excretion (regression)
    ld50_zhu                    # Toxicity (regression)
)

echo "============================================================"
echo "  VIRTUAL NODE ABLATION: PairMixer 10M (toymix)"
echo "  Baseline: ${MODEL_BASE} (GPU ${GPU_BASE})"
echo "  VN:       ${MODEL_VN} (GPU ${GPU_VN})"
echo "  Seeds: ${SEEDS[*]}"
echo "  ADMET tasks: ${#ABLATION_TASKS[@]}"
echo "============================================================"

# ── Phase 1: Pre-training ───────────────────────────────────────────────────
if [[ "${SKIP_PRETRAIN:-0}" != "1" ]]; then
    echo ""
    echo "=== Phase 1: Pre-training ==="

    for SEED_VAL in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Seed ${SEED_VAL}: launching baseline (GPU ${GPU_BASE}) and VN (GPU ${GPU_VN}) <<<"

        SEED=${SEED_VAL} \
            bash "$(dirname "$0")/00_pretrain.sh" "${MODEL_BASE}" "${DATASET}" "${GPU_BASE}" \
            EXTRA_FLAGS="++constants.wandb.tags=['${MODEL_BASE}','pretrain','${DATASET}','ablation_vn','no_vn']" &
        PID_BASE=$!

        SEED=${SEED_VAL} \
            bash "$(dirname "$0")/00_pretrain.sh" "${MODEL_VN}" "${DATASET}" "${GPU_VN}" \
            EXTRA_FLAGS="++constants.wandb.tags=['${MODEL_VN}','pretrain','${DATASET}','ablation_vn','vn_sum']" &
        PID_VN=$!

        echo "  Waiting for baseline (PID ${PID_BASE}) and VN (PID ${PID_VN})..."
        wait ${PID_BASE}
        RC_BASE=$?
        wait ${PID_VN}
        RC_VN=$?

        echo "  Seed ${SEED_VAL} done: baseline=${RC_BASE}, vn=${RC_VN}"
        [[ ${RC_BASE} -ne 0 ]] && echo "  ERROR: baseline failed (exit ${RC_BASE})"
        [[ ${RC_VN} -ne 0 ]] && echo "  ERROR: VN failed (exit ${RC_VN})"
    done

    echo ""
    echo "Pre-training complete."
fi

# ── Phase 2: Fine-tuning ───────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Fine-tuning on ADMET ==="

# Find latest checkpoint for each model+seed combination
find_latest_ckpt() {
    local model=$1
    local seed=$2
    local ckpt_base="models_checkpoints/small-dataset/${model}"

    # Find checkpoint dirs, sort by date descending, pick first with matching seed
    # Since seed is embedded in training, we pick the latest run for now
    local latest_dir
    latest_dir=$(ls -dt "${ckpt_base}"/*/ 2>/dev/null | head -1)

    if [[ -n "${latest_dir}" && -f "${latest_dir}/last.ckpt" ]]; then
        echo "${latest_dir}/last.ckpt"
    else
        echo ""
    fi
}

# Worker: fine-tune one model on all tasks for one seed
run_finetune() {
    local gpu=$1
    local model=$2
    local ckpt=$3
    local seed=$4
    local vn_label=$5

    if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
        echo "WARN: [GPU ${gpu}] No checkpoint for ${model} seed=${seed}, skipping fine-tuning"
        return 1
    fi

    for task in "${ABLATION_TASKS[@]}"; do
        echo "[GPU ${gpu}] ${model} seed=${seed} / ${task}"

        TAGS="['${model}','finetune','admet','${DATASET}','ablation_vn','${vn_label}']"

        CUDA_VISIBLE_DEVICES=${gpu} graphium-train \
            model=${model} \
            accelerator=gpu \
            tasks=admet \
            ++constants.seed=${seed} \
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
            ++trainer.model_checkpoint.dirpath=models_checkpoints/ablation_vn/${model}/seed${seed}/${task}/ \
            ++trainer.model_checkpoint.save_last=False \
            ${EXTRA_FLAGS:-} \
        || echo "WARN: [GPU ${gpu}] ${model}/${task} seed=${seed} failed"

        sleep 1
    done
}

for SEED_VAL in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed ${SEED_VAL}: fine-tuning both conditions in parallel ---"

    CKPT_BASE=$(find_latest_ckpt "${MODEL_BASE}" "${SEED_VAL}")
    CKPT_VN=$(find_latest_ckpt "${MODEL_VN}" "${SEED_VAL}")

    echo "  Baseline ckpt: ${CKPT_BASE:-MISSING}"
    echo "  VN ckpt:       ${CKPT_VN:-MISSING}"

    run_finetune "${GPU_BASE}" "${MODEL_BASE}" "${CKPT_BASE}" "${SEED_VAL}" "no_vn" &
    PID_FT_BASE=$!

    run_finetune "${GPU_VN}" "${MODEL_VN}" "${CKPT_VN}" "${SEED_VAL}" "vn_sum" &
    PID_FT_VN=$!

    wait ${PID_FT_BASE}
    wait ${PID_FT_VN}

    echo "  Seed ${SEED_VAL} fine-tuning done."
done

echo ""
echo "============================================================"
echo "  VIRTUAL NODE ABLATION COMPLETE"
echo "  Results: ${RESULTS_DIR}/experiment_results.csv"
echo "  Filter by wandb_tags containing 'ablation_vn'"
echo "  Compare 'no_vn' vs 'vn_sum' conditions"
echo "============================================================"
