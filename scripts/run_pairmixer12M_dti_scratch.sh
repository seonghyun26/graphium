#!/usr/bin/env bash
# Train PairMixer 12M from scratch on the TDC DTI eval setup (no pretraining).
#
# Default scope: DAVIS + KIBA × random + cold_target × seeds 0..4 = 20 runs on GPU 6.
#
# Usage:
#   bash scripts/run_pairmixer12M_dti_scratch.sh [gpu_id]
#
# Env vars (optional):
#   SUBSETS=DAVIS,KIBA            : comma-separated TDC DTI subsets
#   METHODS=random,cold_target    : comma-separated split methods
#   SEEDS=0,1,2,3,4               : comma-separated seeds (one run per seed)
#   DTI_EVAL_DIR=data/dti-eval    : where the prep script wrote CSVs + splits/
#
# Writes per-run rows to results/experiment_results.csv (W&B project: dti_eval_<subset>).

source "$(dirname "$0")/common.sh"

DEVICE=${1:-6}

SUBSETS=${SUBSETS:-DAVIS,KIBA}
METHODS=${METHODS:-random,cold_target}
SEEDS=${SEEDS:-0,1,2,3,4}
DTI_EVAL_DIR=${DTI_EVAL_DIR:-data/dti-eval}

if [[ ! -d "${DTI_EVAL_DIR}/splits" ]]; then
    echo "ERROR: ${DTI_EVAL_DIR}/splits/ not found."
    echo "       Run scripts/data/dti_eval/01_prepare_dti_eval.py first."
    exit 1
fi

IFS=',' read -ra SUBSET_LIST <<< "${SUBSETS}"
IFS=',' read -ra METHOD_LIST <<< "${METHODS}"
IFS=',' read -ra SEED_LIST <<< "${SEEDS}"

echo "=== PairMixer 12M (scratch) on DTI eval — GPU ${DEVICE} ==="
echo "    subsets : ${SUBSET_LIST[*]}"
echo "    methods : ${METHOD_LIST[*]}"
echo "    seeds   : ${SEED_LIST[*]}"

for subset in "${SUBSET_LIST[@]}"; do
    for method in "${METHOD_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
            csv_path="${DTI_EVAL_DIR}/${subset}.csv"
            split_path="${DTI_EVAL_DIR}/splits/${subset}_${method}_seed${seed}.pt"
            if [[ ! -f "${csv_path}" || ! -f "${split_path}" ]]; then
                echo "WARN: missing ${csv_path} or ${split_path}; skipping."
                continue
            fi

            TAGS="['pairmixer_12M','scratch','dti_eval','${subset}','${method}','seed${seed}']"
            echo "--- ${subset} / ${method} / seed=${seed} ---"

            SEED=${seed} CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
                model=pairmixer_12M \
                accelerator=gpu \
                tasks=dti_eval \
                training=dti_eval \
                $(wandb_flags "${TAGS}") \
                ++constants.task=${subset} \
                ++constants.dti_subset=${subset} \
                ++constants.dti_split_method=${method} \
                ++constants.dti_split_seed=${seed} \
                ++constants.dti_eval_dir=${DTI_EVAL_DIR} \
                ++trainer.model_checkpoint.dirpath=models_checkpoints/dti-eval/scratch/${subset}/${method}/seed${seed}/pairmixer_12M/${TZ:-UTC} \
                ${EXTRA_FLAGS:-} \
            || echo "WARN: ${subset}/${method}/seed=${seed} failed, continuing..."

            sleep 1
        done
    done
done
