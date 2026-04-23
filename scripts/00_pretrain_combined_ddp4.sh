#!/usr/bin/env bash
# 4-GPU DDP variant of 00_pretrain_combined_ddp.sh.
#
# Pre-train a PairMixer backbone on the COMBINED toymix + ESM-C v2 DTI +
# L+M-24 OpenAI-caption + BBBC047 dataset with DDP across FOUR GPUs.
#
# Usage:
#   bash scripts/00_pretrain_combined_ddp4.sh <model> <g0> <g1> <g2> <g3>
#
# Arguments:
#   model            : pairmixer_12M | pairmixer_50M | pairmixer_100M
#   g0..g3           : four CUDA device indices (e.g. 4 5 6 7)
#
# Example:
#   bash scripts/00_pretrain_combined_ddp4.sh pairmixer_50M 4 5 6 7
#
# Forward EXTRA_FLAGS via env for raw Hydra overrides:
#   EXTRA_FLAGS="+ft_monitor=admet_mlp constants.name=my_tag" \
#     bash scripts/00_pretrain_combined_ddp4.sh pairmixer_50M 4 5 6 7
#
# Effective-batch math (per the matching training/model/*_${model}.yaml):
#   pairmixer_50M : per-GPU eff 48  ->  DDP-4 eff 192
#   pairmixer_12M : per-GPU eff 48  ->  DDP-4 eff 192
#   pairmixer_100M: per-GPU eff 48  ->  DDP-4 eff 192
# If you want to hold effective batch closer to the single-GPU anchor
# (e.g. 48 or 96) under 4-way DDP, divide trainer.trainer.accumulate_grad_batches
# by 2 or 4 via EXTRA_FLAGS, e.g.:
#   EXTRA_FLAGS="trainer.trainer.accumulate_grad_batches=3"  (50M -> eff 48)
#
# Logs land at logs/pretrain_combined_ddp4_<model>_<stamp>.log.

set -uo pipefail

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <model> <g0> <g1> <g2> <g3>" >&2
    exit 1
fi
MODEL=$1
G0=$2
G1=$3
G2=$4
G3=$5

case "${MODEL}" in
    pairmixer_12M|pairmixer_50M|pairmixer_100M)
        ;;
    *)
        echo "Error: model '${MODEL}' not supported by this script." >&2
        echo "Known: pairmixer_12M | pairmixer_50M | pairmixer_100M" >&2
        echo "(Others need a matching training/model/toymix_esmc_lpm24_bbbc047_${MODEL}.yaml)" >&2
        exit 1
        ;;
esac

# Duplicate-GPU guard — DDP will hang silently if two ranks map to the same device.
gpus=("${G0}" "${G1}" "${G2}" "${G3}")
uniq_count=$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)
if [[ "${uniq_count}" -ne 4 ]]; then
    echo "Error: duplicate GPU index in ${gpus[*]}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/pretrain_combined_ddp4_${MODEL}_${STAMP}.log"

TAGS="['${MODEL}','pretrain','toymix_esmc_lpm24_bbbc047','ddp4']"
WANDB_FLAGS=$(source "${SCRIPT_DIR}/common.sh"; wandb_flags "${TAGS}")

cat <<EOF
== combined 4-GPU DDP pretrain ==
  model:    ${MODEL}
  gpus:     ${G0},${G1},${G2},${G3}
  tasks:    toymix_esmc_lpm24_bbbc047  (ToyMix + DTI-ESMC-v2 + L+M-24-OpenAI + BBBC047)
  log:      ${LOG}
  extras:   ${EXTRA_FLAGS:-<none>}

Launching in 5 s — Ctrl-C to abort.
EOF
sleep 5

# DDP overrides:
#   devices=4  -> Lightning spawns one process per visible GPU.
#   strategy=ddp_find_unused_parameters_true
#              -> multi-task graphs don't activate every task-head on every
#                 batch, so Lightning would otherwise raise during backward().
CUDA_VISIBLE_DEVICES=${G0},${G1},${G2},${G3} graphium-train \
    model=${MODEL} \
    accelerator=gpu \
    tasks=toymix_esmc_lpm24_bbbc047 \
    training=toymix_esmc_lpm24_bbbc047 \
    architecture=toymix \
    ${WANDB_FLAGS} \
    +trainer.trainer.devices=4 \
    +trainer.trainer.strategy=ddp_find_unused_parameters_true \
    ${EXTRA_FLAGS:-} 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}

echo ""
echo "== Done (exit ${rc}) =="
echo "  log: ${LOG}"
exit ${rc}
