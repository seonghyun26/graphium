#!/usr/bin/env bash
# Benchmark Pairformer model sizes: pretrain on toymix, finetune on 2 ADMET tasks.
#
# Compares: pairformer_small (7M), pairformer_medium (17M),
#           pairformer_large (52M), pairformer_boltz (155M)
#
# Usage:
#   bash scripts/09_benchmark_pairformer_sizes.sh [gpu_id]
#
# Each model: pretrain 50 epochs on toymix → finetune on caco2_wang + solubility_aqsoldb
# Results written to results/pairformer_size_benchmark.tsv

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}
PRETRAIN_EPOCHS=50
FT_TASKS=("caco2_wang" "solubility_aqsoldb")
RESULTS_FILE="${RESULTS_DIR}/pairformer_size_benchmark.tsv"

MODELS=("pairformer_small" "pairformer_medium" "pairformer_large")
# Uncomment to include boltz (very slow):
# MODELS=("pairformer_small" "pairformer_medium" "pairformer_large" "pairformer_boltz")

echo "=== Pairformer Size Benchmark ==="
echo "Models: ${MODELS[*]}"
echo "Pretrain: toymix, ${PRETRAIN_EPOCHS} epochs"
echo "Finetune: ${FT_TASKS[*]}"
echo "GPU: ${DEVICE}"
echo ""

# Write TSV header
mkdir -p "$(dirname "${RESULTS_FILE}")"
echo -e "model\tparams\tpretrain_time_s\tft_task\tft_mae\tft_pearson\tft_time_s" > "${RESULTS_FILE}"

for MODEL in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Model: ${MODEL}"
    echo "============================================================"

    CKPT_DIR="models_checkpoints/benchmark/${MODEL}_$(date +%Y%m%d_%H%M%S)"
    TAGS="['${MODEL}','benchmark','toymix']"

    # ── Phase 1: Pretrain ────────────────────────────────────────────
    echo "--- Pretraining ${MODEL} on toymix (${PRETRAIN_EPOCHS} epochs) ---"
    PT_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=toymix \
        training=toymix \
        architecture=toymix \
        $(wandb_flags "${TAGS}") \
        ++constants.norm=layer_norm \
        ++constants.max_epochs=${PRETRAIN_EPOCHS} \
        ++constants.raise_train_error=true \
        ++trainer.trainer.precision=bf16-mixed \
        ++trainer.trainer.check_val_every_n_epoch=10 \
        ++trainer.model_checkpoint.dirpath=${CKPT_DIR} \
        ++trainer.model_checkpoint.save_last=True \
        ${EXTRA_FLAGS:-} \
    || { echo "WARN: pretrain ${MODEL} failed"; continue; }

    PT_END=$(date +%s)
    PT_TIME=$((PT_END - PT_START))

    # Find checkpoint
    CKPT=$(find "${CKPT_DIR}" -name "last.ckpt" -type f 2>/dev/null | head -1)
    if [[ -z "${CKPT}" ]]; then
        echo "ERROR: No checkpoint for ${MODEL}"
        continue
    fi
    echo "Checkpoint: ${CKPT} (${PT_TIME}s)"

    # Count params from wandb log
    PARAMS=$(grep -r "Total params" "$(dirname "${CKPT}")/../" 2>/dev/null | grep -oP '[\d.]+' | head -1)
    PARAMS=${PARAMS:-"?"}

    # ── Phase 2: Finetune on each ADMET task ─────────────────────────
    for FT_TASK in "${FT_TASKS[@]}"; do
        echo "--- Finetuning ${MODEL} on ${FT_TASK} ---"
        FT_TAGS="['${MODEL}','benchmark','finetune','${FT_TASK}']"
        FT_START=$(date +%s)

        FT_OUTPUT_DIR="outputs/benchmark/${MODEL}_${FT_TASK}_$(date +%Y%m%d_%H%M%S)"

        CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
            model=${MODEL} \
            accelerator=gpu \
            tasks=admet \
            +finetuning=admet \
            $(wandb_flags "${FT_TAGS}") \
            hydra.run.dir=${FT_OUTPUT_DIR} \
            ++constants.raise_train_error=False \
            ++constants.task=${FT_TASK} \
            ++finetuning.task=${FT_TASK} \
            ++datamodule.args.tdc_benchmark_names=${FT_TASK} \
            ++datamodule.args.num_workers=0 \
            ++datamodule.args.featurization_n_jobs=0 \
            ++datamodule.args.processed_graph_data_path=null \
            ++finetuning.pretrained_model=${CKPT} \
            ++finetuning.unfreeze_pretrained_depth=0 \
            ++finetuning.epoch_unfreeze_all=none \
            ++finetuning.finetuning_head.in_dim=256 \
            ++finetuning.finetuning_head.hidden_dims=256 \
            ++finetuning.new_out_dim=256 \
            ++finetuning.finetuning_head.depth=4 \
            ++finetuning.added_depth=4 \
            ++architecture.task_heads.${FT_TASK}.hidden_dims=256 \
            ++trainer.model_checkpoint.save_last=False \
            ${EXTRA_FLAGS:-} \
        || { echo "WARN: finetune ${MODEL}/${FT_TASK} failed"; continue; }

        FT_END=$(date +%s)
        FT_TIME=$((FT_END - FT_START))

        # Extract metrics
        FT_MAE=$(grep "mae/test" "${FT_OUTPUT_DIR}/test_results.yaml" 2>/dev/null | head -1 | awk '{print $2}')
        FT_PEARSON=$(grep "pearson.*test\|pearsonr.*test" "${FT_OUTPUT_DIR}/test_results.yaml" 2>/dev/null | head -1 | awk '{print $2}')
        FT_MAE=${FT_MAE:-"N/A"}
        FT_PEARSON=${FT_PEARSON:-"N/A"}

        echo "  MAE=${FT_MAE}, Pearson=${FT_PEARSON}, Time=${FT_TIME}s"

        # Log to TSV
        echo -e "${MODEL}\t${PARAMS}\t${PT_TIME}\t${FT_TASK}\t${FT_MAE}\t${FT_PEARSON}\t${FT_TIME}" >> "${RESULTS_FILE}"
    done

    echo ""
done

echo "============================================================"
echo "DONE — Results in ${RESULTS_FILE}"
echo "============================================================"
cat "${RESULTS_FILE}" | column -t -s$'\t'
