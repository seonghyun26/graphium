#!/usr/bin/env bash
# Pre-train PairMixer 3M on ALL five pre-training datasets, each with 4-GPU
# DDP. Runs the five legs serially since DDP-4 already saturates the four
# GPUs.
#
# Datasets (in order, default chain — all 4 aux legs ADMET-filtered):
#   1. toymix                        — ToyMix small-dataset only (no ADMET overlap)
#   2. toymix_lpm24_litopenai        — ToyMix + L+M-24 OpenAI captions (ADMET-filtered)
#   3. toymix_bbbc047                — ToyMix + BBBC047 cell-morphology (ADMET-filtered)
#   4. toymix_dti_esmc_v2            — ToyMix + DTI (ESM-C v2, ADMET-filtered)
#
# The combined leg (toymix_dti_esmc_v2_lpm24_litopenai_bbbc047) is still a valid override —
# pass it explicitly via the dataset_csv argument to include it. Note: the
# combined leg's lpm24 sub-table currently still points at the unfiltered
# parquet; update tasks/toymix_dti_esmc_v2_lpm24_litopenai_bbbc047.yaml if you re-enable it.
#
# ADMET filtering removes rows whose canonical SMILES match the union of all
# 22 ADMET task splits (~46k SMILES, datacache/admet_smiles/admet_all_smiles.json).
# Per-leg filter ratios: esmc/bbbc047 done upstream; lpm24 = 1,740/129,634 (1.34%).
#
# Pre-training scheme: MiniMol-style — lr=4e-4, WarmUpLinearLR, 5 warmup
# epochs, 100 max epochs (LR set in the pairmixer_3M model config).
#
# Usage:
#   bash scripts/00_pretrain_all_3M_ddp4.sh <g0> <g1> <g2> <g3> [datasets]
#
# Arguments:
#   g0..g3   : four CUDA device indices (e.g. 4 5 6 7)
#   datasets : optional comma-separated subset of the chain. Default:
#                toymix,toymix_dti_esmc_v2,toymix_lpm24_litopenai,toymix_bbbc047,toymix_dti_esmc_v2_lpm24_litopenai_bbbc047
#              Use this to re-run a single leg after an OOM or crash.
#
# Examples:
#   bash scripts/00_pretrain_all_3M_ddp4.sh 4 5 6 7
#   bash scripts/00_pretrain_all_3M_ddp4.sh 4 5 6 7 toymix_bbbc047
#   bash scripts/00_pretrain_all_3M_ddp4.sh 4 5 6 7 toymix,toymix_dti_esmc_v2
#
# Forward EXTRA_FLAGS via env for raw Hydra overrides:
#   EXTRA_FLAGS="constants.max_epochs=50" bash scripts/00_pretrain_all_3M_ddp4.sh 4 5 6 7
#
# Skip the per-leg cache warmup (single-process prepare_data() pass) once
# the cache is hot:
#   WARMUP_SKIP=1 bash scripts/00_pretrain_all_3M_ddp4.sh 4 5 6 7
#
# Logs land at logs/pretrain_all_3M_ddp4_<stamp>/<dataset>.log.

set -uo pipefail

# ---- Argument parsing --------------------------------------------------------
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <g0> <g1> <g2> <g3> [dataset_csv]" >&2
    exit 1
fi
G0=$1
G1=$2
G2=$3
G3=$4
DATASET_CSV=${5:-toymix,toymix_lpm24_litopenai,toymix_bbbc047,toymix_dti_esmc_v2}

# Duplicate-GPU guard — DDP will hang silently if two ranks map to the same device.
gpus=("${G0}" "${G1}" "${G2}" "${G3}")
uniq_count=$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)
if [[ "${uniq_count}" -ne 4 ]]; then
    echo "Error: duplicate GPU index in ${gpus[*]}" >&2
    exit 1
fi

IFS=',' read -ra DATASETS <<< "${DATASET_CSV}"
for ds in "${DATASETS[@]}"; do
    case "${ds}" in
        toymix|toymix_dti_esmc_v2|toymix_lpm24_litopenai|toymix_bbbc047|toymix_dti_esmc_v2_lpm24_litopenai_bbbc047) ;;
        *) echo "Error: unknown dataset '${ds}'" >&2; exit 1 ;;
    esac
done

# ---- Paths / logs ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_DIR}/logs/pretrain_all_3M_ddp4_${STAMP}"
mkdir -p "${LOG_DIR}"

MODEL=pairmixer_3M

# ---- Dataset → (tasks, training, architecture) groups ------------------------
# Hydra auto-resolves training/accelerator/${training}_${accel} and
# training/model/${training}_${MODEL}, so we only set the three top-level groups.
# architecture=toymix for all five legs (small-dataset axis).
groups_for() {
    case "$1" in
        toymix)
            TASKS=toymix
            TRAINING=toymix
            ARCH=toymix
            ;;
        toymix_dti_esmc_v2)
            TASKS=toymix_dti_esmc_v2
            TRAINING=toymix_dti_esmc_v2
            ARCH=toymix
            ;;
        toymix_lpm24_litopenai)
            TASKS=toymix_lpm24_litopenai
            TRAINING=toymix_lpm24_litopenai
            ARCH=toymix
            ;;
        toymix_bbbc047)
            TASKS=toymix_bbbc047
            TRAINING=toymix_bbbc047
            ARCH=toymix
            ;;
        toymix_dti_esmc_v2_lpm24_litopenai_bbbc047)
            TASKS=toymix_dti_esmc_v2_lpm24_litopenai_bbbc047
            TRAINING=toymix_dti_esmc_v2_lpm24_litopenai_bbbc047
            ARCH=toymix
            ;;
    esac
}

# ---- Build the launch command for one dataset --------------------------------
build_cmd() {
    local dataset=$1
    groups_for "${dataset}"
    local tags="['${MODEL}','pretrain','${dataset}','ddp4','minimol_lr']"
    local wandb_flags
    # Collapse newlines: wandb_flags() emits one flag per line, but embedding
    # a multi-line value inside the heredoc breaks the `\`-line-continuation
    # chain (interior lines lose the trailing `\`), causing eval to drop every
    # flag after `${wandb_flags}` — including +trainer.trainer.devices=4.
    wandb_flags=$(source "${SCRIPT_DIR}/common.sh"; wandb_flags "${tags}" | tr '\n' ' ')

    cat <<EOF
CUDA_VISIBLE_DEVICES=${G0},${G1},${G2},${G3} graphium-train \\
    model=${MODEL} \\
    accelerator=gpu \\
    tasks=${TASKS} \\
    training=${TRAINING} \\
    architecture=${ARCH} \\
    ${wandb_flags} \\
    +trainer.trainer.devices=4 \\
    +trainer.trainer.strategy=ddp_find_unused_parameters_true \\
    +trainer.trainer.ddp_timeout_hours=6 \\
    ${EXTRA_FLAGS:-}
EOF
}

# ---- Build the cache-warmup command for one dataset --------------------------
# Single-process pre-pass that runs prepare_data() and exits — populates
# featurization + ESM-C cache so DDP ranks rendezvous within the TCPStore
# timeout. Skip with WARMUP_SKIP=1.
build_warmup_cmd() {
    local dataset=$1
    groups_for "${dataset}"
    cat <<EOF
CUDA_VISIBLE_DEVICES=${G0} graphium-train \\
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
echo "== PairMixer 3M all-pretrain DDP-4 plan =="
echo "  model:    ${MODEL}"
echo "  gpus:     ${G0},${G1},${G2},${G3}"
echo "  datasets: ${DATASETS[*]}"
echo "  log dir:  ${LOG_DIR}"
echo "  scheme:   MiniMol-style (lr=4e-4, WarmUpLinearLR, 5 warmup ep, 100 max ep)"
echo ""
for i in "${!DATASETS[@]}"; do
    echo "--- [$((i+1))/${#DATASETS[@]}] ${DATASETS[i]} ---"
    build_cmd "${DATASETS[i]}"
    echo ""
done
echo "Starting in 5 s — Ctrl-C to abort."
sleep 5

# ---- Run each leg serially ---------------------------------------------------
declare -a STATUS
RC_GLOBAL=0

for i in "${!DATASETS[@]}"; do
    ds=${DATASETS[i]}
    log="${LOG_DIR}/${ds}.log"
    echo ""
    echo "============================================================"
    echo "  [$((i+1))/${#DATASETS[@]}] ${ds} -> ${log}"
    echo "============================================================"

    # Cache warmup before DDP.
    if [[ "${WARMUP_SKIP:-0}" != "1" ]]; then
        echo ">> [warmup] ${ds} on GPU ${G0} ..."
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
    printf "  %-32s %s\n" "${DATASETS[i]}" "${STATUS[i]}"
done
echo "  log dir: ${LOG_DIR}"
exit ${RC_GLOBAL}
