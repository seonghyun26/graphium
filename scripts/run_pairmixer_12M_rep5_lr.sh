#!/usr/bin/env bash
# Run the 5 representative ADMET tasks for PairMixer 12M LR/scheduler checks.

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE=${1:-6}
CKPT=${2:?Usage: $0 <gpu_id> <checkpoint>}
MODEL=${MODEL:-pairmixer_12M}
GRAPHIUM_TRAIN=${GRAPHIUM_TRAIN:-/home/shpark/.conda/envs/graphium/bin/graphium-train}
WANDB_ENTITY=${WANDB_ENTITY:-eddy26}
WANDB_PROJECT=${WANDB_PROJECT:-graphium}
RESULTS_DIR=${RESULTS_DIR:-$(pwd)/results}
RESULT_ROOT=${RESULT_ROOT:-models_checkpoints/admet-representative}

REP_TASKS=(
  caco2_wang
  bbb_martins
  cyp2d6_veith
  clearance_microsome_az
  ld50_zhu
)

run_one() {
  local label=$1
  local finetune_cfg=$2
  local lr=$3
  local task=$4

  CUDA_VISIBLE_DEVICES=${DEVICE} "${GRAPHIUM_TRAIN}" \
    model=${MODEL} \
    accelerator=gpu \
    tasks=admet \
    ++constants.seed=0 \
    ++constants.wandb.entity=${WANDB_ENTITY} \
    ++constants.wandb.project=${WANDB_PROJECT} \
    ++constants.wandb.tags="['${MODEL}','finetune','admet','toymix','rep5','${label}']" \
    ++constants.results_csv_dir=${RESULTS_DIR} \
    ++constants.task=${task} \
    +finetuning=${finetune_cfg} \
    ++finetuning.task=${task} \
    ++datamodule.args.tdc_benchmark_names=${task} \
    ++finetuning.pretrained_model=${CKPT} \
    ++finetuning.sub_module_from_pretrained=zinc \
    ++finetuning.unfreeze_pretrained_depth=0 \
    ++finetuning.epoch_unfreeze_all=none \
    ++finetuning.finetuning_head.in_dim=256 \
    ++finetuning.finetuning_head.hidden_dims=256 \
    ++finetuning.new_out_dim=256 \
    ++finetuning.finetuning_head.depth=4 \
    ++finetuning.added_depth=4 \
    ++architecture.task_heads.${task}.hidden_dims=256 \
    ++predictor.optim_kwargs.lr=${lr} \
    ++constants.name=finetuning_${task}_${MODEL}_${label} \
    ++trainer.model_checkpoint.dirpath=${RESULT_ROOT}/toymix/${MODEL}/${label}/${task}/
}

for task in "${REP_TASKS[@]}"; do
  run_one baseline_lr1e4 admet 1e-4 "${task}"
done

for task in "${REP_TASKS[@]}"; do
  run_one warmup_cosine_lr1e4 admet_cosine 1e-4 "${task}"
done

for task in "${REP_TASKS[@]}"; do
  run_one baseline_lr5e5 admet 5e-5 "${task}"
done

for task in "${REP_TASKS[@]}"; do
  run_one baseline_lr2e4 admet 2e-4 "${task}"
done
