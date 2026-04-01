#!/usr/bin/env bash
# Benchmark speed and memory: Pairformer vs PairMixer (boltz config).
#
# Runs a short training loop (3 epochs, 50 batches) on toymix for each model
# and records wall-clock time, peak GPU memory, and training loss.
#
# Usage:
#   bash scripts/11_benchmark_pairformer_vs_pairmixer.sh [gpu_id]
#
# Results written to results/pairformer_vs_pairmixer_benchmark.tsv

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}
RESULTS_FILE="${RESULTS_DIR}/pairformer_vs_pairmixer_benchmark.tsv"

MODELS=("pairformer_boltz" "pairmixer_boltz")
MAX_EPOCHS=3
MAX_STEPS=50
BATCH_SIZE=16

echo "=== Pairformer vs PairMixer Speed/Memory Benchmark ==="
echo "Models:     ${MODELS[*]}"
echo "Epochs:     ${MAX_EPOCHS}"
echo "Max steps:  ${MAX_STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "GPU:        ${DEVICE}"
echo ""

mkdir -p "$(dirname "${RESULTS_FILE}")"
echo -e "model\tepochs\tmax_steps\tbatch_size\twall_time_s\tpeak_mem_MiB\ttrain_loss\tqm9_mae\tzinc_mae\ttox21_auroc" > "${RESULTS_FILE}"

for MODEL in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Benchmarking: ${MODEL}"
    echo "============================================================"

    TAGS="['${MODEL}','speed_benchmark','toymix']"
    OUTPUT_DIR="outputs/benchmark/${MODEL}_speed_$(date +%Y%m%d_%H%M%S)"

    # Clear GPU memory before run
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    START_TIME=$(date +%s)

    CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
        model=${MODEL} \
        accelerator=gpu \
        tasks=toymix \
        training=toymix \
        architecture=toymix \
        hydra.run.dir=${OUTPUT_DIR} \
        $(wandb_flags "${TAGS}") \
        ++datamodule.args.batch_size_training=${BATCH_SIZE} \
        ++datamodule.args.batch_size_inference=${BATCH_SIZE} \
        ++datamodule.args.num_workers=0 \
        ++datamodule.args.featurization_n_jobs=0 \
        ++trainer.trainer.max_epochs=${MAX_EPOCHS} \
        ++trainer.trainer.limit_train_batches=${MAX_STEPS} \
        ++trainer.trainer.limit_val_batches=10 \
        ++trainer.trainer.limit_test_batches=10 \
        ++trainer.trainer.precision=bf16-mixed \
        ++trainer.model_checkpoint.save_last=False \
        ++constants.raise_train_error=True \
        ${EXTRA_FLAGS:-} \
    || { echo "WARN: ${MODEL} failed"; echo -e "${MODEL}\t${MAX_EPOCHS}\t${MAX_STEPS}\t${BATCH_SIZE}\tFAIL\tFAIL\tFAIL\tFAIL\tFAIL\tFAIL" >> "${RESULTS_FILE}"; continue; }

    END_TIME=$(date +%s)
    WALL_TIME=$((END_TIME - START_TIME))

    # Peak GPU memory
    PEAK_MEM=$(nvidia-smi -i "${DEVICE}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    PEAK_MEM="${PEAK_MEM:-N/A}"

    # Extract metrics from test results
    QM9_MAE=$(grep "graph_qm9/mae/test" "${OUTPUT_DIR}/test_results.yaml" 2>/dev/null | awk '{print $2}' || echo "N/A")
    ZINC_MAE=$(grep "graph_zinc/mae/test" "${OUTPUT_DIR}/test_results.yaml" 2>/dev/null | awk '{print $2}' || echo "N/A")
    TOX21_AUROC=$(grep "graph_tox21/auroc/test" "${OUTPUT_DIR}/test_results.yaml" 2>/dev/null | awk '{print $2}' || echo "N/A")
    TRAIN_LOSS=$(grep "loss/test" "${OUTPUT_DIR}/test_results.yaml" 2>/dev/null | head -1 | awk '{print $2}' || echo "N/A")

    echo ""
    echo "  Wall time:    ${WALL_TIME}s"
    echo "  Peak mem:     ${PEAK_MEM} MiB"
    echo "  Train loss:   ${TRAIN_LOSS}"
    echo "  QM9 MAE:      ${QM9_MAE}"
    echo "  ZINC MAE:     ${ZINC_MAE}"
    echo "  Tox21 AUROC:  ${TOX21_AUROC}"
    echo ""

    echo -e "${MODEL}\t${MAX_EPOCHS}\t${MAX_STEPS}\t${BATCH_SIZE}\t${WALL_TIME}\t${PEAK_MEM}\t${TRAIN_LOSS}\t${QM9_MAE}\t${ZINC_MAE}\t${TOX21_AUROC}" >> "${RESULTS_FILE}"

    sleep 2
done

echo "============================================================"
echo "DONE — Results:"
echo "============================================================"
column -t -s$'\t' "${RESULTS_FILE}"
