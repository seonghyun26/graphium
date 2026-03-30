#!/usr/bin/env bash
# Benchmark Pairformer model sizes: train from scratch on 5 ADMET tasks.
#
# Compares: pairformer (5M), pairformer_small (7M),
#           pairformer_medium (17M), pairformer_large (52M)
#
# Usage:
#   bash scripts/09_benchmark_pairformer_sizes.sh [gpu_id]
#
# Results written to results/pairformer_size_benchmark.tsv

source "$(dirname "$0")/common.sh"

DEVICE=${1:-${DEVICE}}
RESULTS_FILE="${RESULTS_DIR}/pairformer_size_benchmark.tsv"

MODELS=("pairformer" "pairformer_small" "pairformer_medium" "pairformer_large")

# 5 tasks: 3 regression + 2 classification (diverse ADMET categories)
BENCH_TASKS=("caco2_wang" "solubility_aqsoldb" "half_life_obach" "pgp_broccatelli" "herg")

echo "=== Pairformer Size Benchmark (scratch ADMET) ==="
echo "Models: ${MODELS[*]}"
echo "Tasks:  ${BENCH_TASKS[*]}"
echo "GPU:    ${DEVICE}"
echo ""

# Write TSV header only if file doesn't exist or is empty
mkdir -p "$(dirname "${RESULTS_FILE}")"
if [[ ! -s "${RESULTS_FILE}" ]]; then
    echo -e "model\ttask\tmae_or_auroc\tpearson\ttime_s" > "${RESULTS_FILE}"
fi

for MODEL in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Model: ${MODEL}"
    echo "============================================================"

    for task in "${BENCH_TASKS[@]}"; do
        echo "--- ${MODEL} / ${task} ---"
        TAGS="['${MODEL}','scratch_benchmark','${task}']"
        FT_START=$(date +%s)

        OUTPUT_DIR="outputs/benchmark/${MODEL}_${task}_$(date +%Y%m%d_%H%M%S)"

        CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
            model=${MODEL} \
            accelerator=gpu \
            tasks=admet \
            $(wandb_flags "${TAGS}") \
            hydra.run.dir=${OUTPUT_DIR} \
            ++constants.raise_train_error=False \
            ++constants.task=${task} \
            ++datamodule.args.tdc_benchmark_names=${task} \
            ++constants.norm=layer_norm \
            ++datamodule.args.num_workers=0 \
            ++datamodule.args.featurization_n_jobs=0 \
            ++datamodule.args.processed_graph_data_path=null \
            ++datamodule.args.batch_size_training=32 \
            ++trainer.trainer.precision=bf16-mixed \
            ++architecture.task_heads.${task}.hidden_dims=256 \
            ++architecture.task_heads.${task}.depth=4 \
            ++trainer.model_checkpoint.save_last=False \
            ${EXTRA_FLAGS:-} \
        || { echo "WARN: ${MODEL}/${task} failed"; echo -e "${MODEL}\t${task}\tFAIL\tFAIL\t0" >> "${RESULTS_FILE}"; continue; }

        FT_END=$(date +%s)
        FT_TIME=$((FT_END - FT_START))

        # Extract metrics (mae for regression, auroc for classification)
        MAE=$(grep "mae/test" "${OUTPUT_DIR}/test_results.yaml" 2>/dev/null | head -1 | awk '{print $2}' || true)
        AUROC=$(grep "auroc/test" "${OUTPUT_DIR}/test_results.yaml" 2>/dev/null | head -1 | awk '{print $2}' || true)
        PEARSON=$(grep -E "pearson.*test|pearsonr.*test" "${OUTPUT_DIR}/test_results.yaml" 2>/dev/null | head -1 | awk '{print $2}' || true)

        METRIC=${MAE:-${AUROC:-"N/A"}}
        PEARSON=${PEARSON:-"N/A"}
        # Replace YAML .nan with N/A
        [[ "${METRIC}" == ".nan" ]] && METRIC="N/A"
        [[ "${PEARSON}" == ".nan" ]] && PEARSON="N/A"

        echo "  metric=${METRIC}, pearson=${PEARSON}, time=${FT_TIME}s"
        echo -e "${MODEL}\t${task}\t${METRIC}\t${PEARSON}\t${FT_TIME}" >> "${RESULTS_FILE}"

        sleep 1
    done
    echo ""
done

echo "============================================================"
echo "DONE — Results:"
echo "============================================================"
column -t -s$'\t' "${RESULTS_FILE}"
