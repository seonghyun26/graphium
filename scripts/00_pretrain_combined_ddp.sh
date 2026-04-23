#!/usr/bin/env bash
# Pre-train a PairMixer backbone on the COMBINED toymix + ESM-C v2 DTI +
# L+M-24 OpenAI-caption + BBBC047 dataset with DDP across two GPUs.
# Companion to 00_pretrain_3aux_ddp.sh, which trains the three aux datasets
# separately; this one trains them jointly in a single multi-task run.
#
# Usage:
#   bash scripts/00_pretrain_combined_ddp.sh <model> <gpu_a> <gpu_b>
#
# Arguments:
#   model    : pairmixer_12M | pairmixer_50M | pairmixer_100M
#   gpu_a/b  : two CUDA device indices (e.g. 6 7)
#
# Examples:
#   bash scripts/00_pretrain_combined_ddp.sh pairmixer_50M 6 7
#   bash scripts/00_pretrain_combined_ddp.sh pairmixer_12M 4 5
#
# Forward EXTRA_FLAGS via env to append raw Hydra overrides:
#   EXTRA_FLAGS="+ft_monitor=admet_mlp constants.name=my_tag" \
#     bash scripts/00_pretrain_combined_ddp.sh pairmixer_50M 6 7
#
# Logs land at logs/pretrain_combined_ddp_<model>_<stamp>.log. Checkpoints
# land under models_checkpoints/toymix_esmc_lpm24_bbbc047/<model>/<stamp>/.

set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <model> <gpu_a> <gpu_b>" >&2
    exit 1
fi
MODEL=$1
GPU_A=$2
GPU_B=$3

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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/pretrain_combined_ddp_${MODEL}_${STAMP}.log"

# Pull wandb_flags helper from common.sh without polluting our namespace.
TAGS="['${MODEL}','pretrain','toymix_esmc_lpm24_bbbc047','ddp']"
WANDB_FLAGS=$(source "${SCRIPT_DIR}/common.sh"; wandb_flags "${TAGS}")

# Print the plan before launching so accidental typos are easy to catch.
cat <<EOF
== combined DDP pretrain ==
  model:    ${MODEL}
  gpus:     ${GPU_A},${GPU_B}
  tasks:    toymix_esmc_lpm24_bbbc047  (ToyMix + DTI-ESMC-v2 + L+M-24-OpenAI + BBBC047)
  log:      ${LOG}
  extras:   ${EXTRA_FLAGS:-<none>}

Launching in 5 s — Ctrl-C to abort.
EOF
sleep 5

# DDP overrides:
#   devices=2  -> Lightning spawns one process per GPU.
#   strategy=ddp_find_unused_parameters_true
#              -> multi-task graphs don't activate every task-head on every
#                 batch, so Lightning would otherwise raise during backward().
CUDA_VISIBLE_DEVICES=${GPU_A},${GPU_B} graphium-train \
    model=${MODEL} \
    accelerator=gpu \
    tasks=toymix_esmc_lpm24_bbbc047 \
    training=toymix_esmc_lpm24_bbbc047 \
    architecture=toymix \
    ${WANDB_FLAGS} \
    +trainer.trainer.devices=2 \
    +trainer.trainer.strategy=ddp_find_unused_parameters_true \
    ${EXTRA_FLAGS:-} 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}

echo ""
echo "== Done (exit ${rc}) =="
echo "  log: ${LOG}"
exit ${rc}
