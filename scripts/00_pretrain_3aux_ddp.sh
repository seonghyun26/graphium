#!/usr/bin/env bash
# Pre-train a PairMixer backbone on three auxiliary-modality pools
# (ToyMix + DTI-ESMC-v2, ToyMix + BBBC047, ToyMix + L+M-24-OpenAI-captions),
# each with DDP across two GPUs. Runs serially — DDP uses both GPUs per
# dataset, so parallel-datasets would oversubscribe.
#
# Usage:
#   bash scripts/00_pretrain_3aux_ddp.sh <model> <gpu_a> <gpu_b> [datasets]
#
# Arguments:
#   model      : pairmixer_12M | pairmixer_50M
#                (other pairmixer sizes work too, but this script only ships
#                 training/model/*_pairmixer_{12M,50M}.yaml for the 3 aux datasets)
#   gpu_a/b    : two CUDA device indices (e.g. 6 7)
#   datasets   : optional comma-separated subset of the chain. Default:
#                  toymix_dti_esmc_v2,toymix_bbbc047,toymix_lpm24_litopenai
#                Use this to re-run a single leg after an OOM or crash.
#
# Examples:
#   bash scripts/00_pretrain_3aux_ddp.sh pairmixer_12M 6 7
#   bash scripts/00_pretrain_3aux_ddp.sh pairmixer_50M 4 5
#   bash scripts/00_pretrain_3aux_ddp.sh pairmixer_50M 0 1 toymix_bbbc047
#
# The script:
#   - prints the three launch commands up-front so you can Ctrl-C if wrong;
#   - writes a per-dataset log under logs/pretrain_3aux_ddp_<model>_<stamp>/;
#   - exits non-zero on the first failure but always summarises which legs
#     completed and which didn't.

set -uo pipefail

# ---- Argument parsing --------------------------------------------------------
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <model> <gpu_a> <gpu_b> [dataset_csv]" >&2
    exit 1
fi
MODEL=$1
GPU_A=$2
GPU_B=$3
DATASET_CSV=${4:-toymix_dti_esmc_v2,toymix_bbbc047,toymix_lpm24_litopenai}

# Validate model: every leg needs a matching training/model/*_pairmixer_*.yaml,
# so restrict to the sizes this script ships configs for.
case "${MODEL}" in
    pairmixer_12M|pairmixer_50M)
        ;;
    *)
        echo "Error: model '${MODEL}' not supported by this script (need pairmixer_12M or pairmixer_50M)." >&2
        exit 1
        ;;
esac

IFS=',' read -ra DATASETS <<< "${DATASET_CSV}"
for ds in "${DATASETS[@]}"; do
    case "${ds}" in
        toymix_dti_esmc_v2|toymix_bbbc047|toymix_lpm24_litopenai) ;;
        *) echo "Error: unknown dataset '${ds}'" >&2; exit 1 ;;
    esac
done

# ---- Paths / logs ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_DIR}/logs/pretrain_3aux_ddp_${MODEL}_${STAMP}"
mkdir -p "${LOG_DIR}"

# ---- Dataset → (tasks, training, training/model) groups ---------------------
# Hydra auto-resolves training/accelerator/${training}_${accel} and
# training/model/${training}_${model}, so we only need to set the three
# top-level groups. architecture=toymix for all three (small-dataset axis).
groups_for() {
    case "$1" in
        toymix_dti_esmc_v2)
            TASKS=toymix_dti_esmc_v2
            TRAINING=toymix_dti_esmc_v2
            ARCH=toymix
            ;;
        toymix_bbbc047)
            TASKS=toymix_bbbc047
            TRAINING=toymix_bbbc047
            ARCH=toymix
            ;;
        toymix_lpm24_litopenai)
            TASKS=toymix_lpm24_litopenai
            TRAINING=toymix_lpm24_litopenai
            ARCH=toymix
            ;;
    esac
}

# ---- Build the launch command for one dataset --------------------------------
build_cmd() {
    local dataset=$1
    groups_for "${dataset}"
    local tags="['${MODEL}','pretrain','${dataset}','ddp']"
    # Wandb flags helper is in common.sh — source it lazily so we don't
    # re-export globals that belong to its own call chain.
    local wandb_flags
    wandb_flags=$(source "${SCRIPT_DIR}/common.sh"; wandb_flags "${tags}")

    # DDP overrides:
    #   devices=2                       -> Lightning spawns one process per GPU
    #   strategy=ddp_find_unused_parameters_true
    #                                    -> multi-task graphs often leave task-specific
    #                                       params unused on some steps; without this
    #                                       Lightning raises during backward().
    #
    # CUDA_VISIBLE_DEVICES maps the two requested GPUs to logical ids 0,1 for
    # Lightning's device index.
    cat <<EOF
CUDA_VISIBLE_DEVICES=${GPU_A},${GPU_B} graphium-train \\
    model=${MODEL} \\
    accelerator=gpu \\
    tasks=${TASKS} \\
    training=${TRAINING} \\
    architecture=${ARCH} \\
    ${wandb_flags} \\
    +trainer.trainer.devices=2 \\
    +trainer.trainer.strategy=ddp_find_unused_parameters_true \\
    ${EXTRA_FLAGS:-}
EOF
}

# ---- Build the cache-warmup command for one dataset --------------------------
# Single-process pre-pass that runs prepare_data() and exits, so the DDP
# ranks below hit a populated featurization cache and rendezvous within the
# TCPStore timeout. Skip with WARMUP_SKIP=1.
build_warmup_cmd() {
    local dataset=$1
    groups_for "${dataset}"
    cat <<EOF
CUDA_VISIBLE_DEVICES=${GPU_A} graphium-train \\
    model=${MODEL} \\
    accelerator=gpu \\
    tasks=${TASKS} \\
    training=${TRAINING} \\
    architecture=${ARCH} \\
    +trainer.trainer.devices=1 \\
    +prepare_data_only=true \\
    ${EXTRA_FLAGS:-}
EOF
}

# ---- Plan preview ------------------------------------------------------------
echo "== 3-aux DDP pretrain plan =="
echo "  model:    ${MODEL}"
echo "  gpus:     ${GPU_A},${GPU_B}"
echo "  datasets: ${DATASETS[*]}"
echo "  log dir:  ${LOG_DIR}"
echo ""
for i in "${!DATASETS[@]}"; do
    echo "--- [$((i+1))/${#DATASETS[@]}] ${DATASETS[i]} ---"
    build_cmd "${DATASETS[i]}"
    echo ""
done
echo "Starting in 5 s — Ctrl-C to abort."
sleep 5

# ---- Run each leg serially ---------------------------------------------------
# We mark successes/failures and always print a summary at the end.
declare -a STATUS
RC_GLOBAL=0

for i in "${!DATASETS[@]}"; do
    ds=${DATASETS[i]}
    log="${LOG_DIR}/${ds}.log"
    echo ""
    echo "============================================================"
    echo "  [$((i+1))/${#DATASETS[@]}] ${ds} -> ${log}"
    echo "============================================================"
    # Cache warmup before DDP — single-process pass that exits after
    # prepare_data(). Set WARMUP_SKIP=1 to skip when the cache is hot.
    if [[ "${WARMUP_SKIP:-0}" != "1" ]]; then
        echo ">> [warmup] ${ds} on GPU ${GPU_A} ..."
        warmup_cmd=$(build_warmup_cmd "${ds}")
        (eval "${warmup_cmd}") 2>&1 | tee -a "${log}"
        wrc=${PIPESTATUS[0]}
        if [[ ${wrc} -ne 0 ]]; then
            echo "!! warmup failed for ${ds} (exit ${wrc}); skipping DDP run." >&2
            STATUS[i]="WARMUP_FAIL(${wrc})"
            RC_GLOBAL=${wrc}
            continue
        fi
        echo ">> [warmup] done."
    fi
    # Eval the command so the heredoc's line continuations collapse properly.
    cmd=$(build_cmd "${ds}")
    (eval "${cmd}") 2>&1 | tee -a "${log}"
    rc=${PIPESTATUS[0]}
    if [[ ${rc} -eq 0 ]]; then
        STATUS[i]="OK"
    else
        STATUS[i]="FAIL(${rc})"
        RC_GLOBAL=${rc}
    fi
done

# ---- Summary -----------------------------------------------------------------
echo ""
echo "== Summary =="
for i in "${!DATASETS[@]}"; do
    printf "  %-30s %s\n" "${DATASETS[i]}" "${STATUS[i]}"
done
echo "  log dir: ${LOG_DIR}"
exit ${RC_GLOBAL}
