#!/usr/bin/env bash
# caco2_wang  side-by-side  scratch  vs  ESM-Cv2-pretrained PairMixer-12M.
#
# Runs N seeds per leg, in parallel across two GPUs, with GRAPHIUM_DUMP_PREDS=1
# so each run drops a predictions.pt that the MiniMol-style aggregator can
# consume. After all seeds finish, calls aggregate_admet_ensemble.py for both
# legs to produce results/experiment_results_ensemble.csv rows.
#
# Why a wrapper instead of the existing 00_finetune_admet_ensemble.sh?
# The existing script runs SEEDS x ALL_TASKS sequentially on ONE GPU. This
# wrapper specialises to one task and parallelises the two pretrain legs
# across two GPUs (since you have idle ones). It still uses 00_scratch_admet.sh
# / 00_finetune_admet.sh internally — same code path as the bigger sweep.
#
# Usage:
#   bash scripts/01_compare_caco2_scratch_vs_esmcv2.sh [ckpt] [gpu_scratch] [gpu_pretrain] [seeds]
#
# Arguments:
#   ckpt         : ESM-Cv2-pretrained PairMixer-12M last.ckpt path. Defaults
#                  to the most recently modified
#                  models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/*/last.ckpt
#   gpu_scratch  : CUDA index for the scratch leg                  (default: 5)
#   gpu_pretrain : CUDA index for the pretrained-finetune leg      (default: 7)
#   seeds        : space-separated seed list                       (default: "0 1 2 3 4")
#
# Examples:
#   bash scripts/01_compare_caco2_scratch_vs_esmcv2.sh
#   bash scripts/01_compare_caco2_scratch_vs_esmcv2.sh \
#        models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/2026-04-18_11-23-58_20260418_112358/last.ckpt \
#        5 7 "0 1 2 3 4"
#   # 10-seed full ensemble:
#   bash scripts/01_compare_caco2_scratch_vs_esmcv2.sh '' 5 7 "0 1 2 3 4 5 6 7 8 9"
#
# Outputs:
#   logs/compare_caco2_{scratch,esmcv2}_seed<N>_<stamp>.log     per-seed log
#   outputs/<date>/<time>/predictions.pt                        per-seed dump
#   results/experiment_results.csv                              per-seed metric rows (always)
#   results/experiment_results_ensemble.csv                     ensemble rows (after aggregator)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

DEFAULT_CKPT_GLOB="models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/*/last.ckpt"

CKPT=${1:-}
GPU_SCRATCH=${2:-5}
GPU_PRETRAIN=${3:-7}
SEEDS=${4:-"0 1 2 3 4"}

# ── Resolve checkpoint ───────────────────────────────────────────────────────
if [[ -z "${CKPT}" ]]; then
    CKPT=$(ls -1t ${DEFAULT_CKPT_GLOB} 2>/dev/null | head -1 || true)
    if [[ -z "${CKPT}" ]]; then
        echo "ERROR: no ESM-Cv2 12M checkpoint found under ${DEFAULT_CKPT_GLOB}" >&2
        echo "       Pass one explicitly as the first argument." >&2
        exit 1
    fi
fi
if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

# Pretrain label that the aggregator will key by (must match what
# _infer_pretrain in scripts/aggregate_admet_ensemble.py extracts from
# the ckpt path: parent-of-grandparent-of-last.ckpt = "toymix-dti-esmc-v2").
PRETRAIN_LABEL="toymix-dti-esmc-v2"

# ── GPU sanity ───────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    n_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    if [[ "${GPU_SCRATCH}" -ge "${n_gpus}" || "${GPU_PRETRAIN}" -ge "${n_gpus}" ]]; then
        echo "ERROR: requested GPU index out of range (have ${n_gpus} GPUs)." >&2
        exit 1
    fi
    if [[ "${GPU_SCRATCH}" -eq "${GPU_PRETRAIN}" ]]; then
        echo "ERROR: scratch and pretrain GPU must differ (got ${GPU_SCRATCH} == ${GPU_PRETRAIN})." >&2
        exit 1
    fi
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Pre-flight summary so a typo in seeds/CKPT/GPUs is obvious before we burn time.
read -ra SEED_ARR <<< "${SEEDS}"
cat <<EOF
== caco2_wang  scratch  vs  ESM-Cv2  (PairMixer-12M, MiniMol-style ensemble) ==
  scratch    -> GPU ${GPU_SCRATCH}
  pretrained -> GPU ${GPU_PRETRAIN}
  checkpoint :  ${CKPT}
  seeds      :  ${SEED_ARR[*]}      (${#SEED_ARR[@]} runs per leg)
  pretrain label (used by aggregator): ${PRETRAIN_LABEL}
  log dir    :  ${LOG_DIR}/

  GRAPHIUM_DUMP_PREDS=1 is set on every leg, so each run writes
  predictions.pt under its Hydra rundir (outputs/<date>/<time>/).

Launching in 5 s — Ctrl-C to abort.
EOF
sleep 5

# ── Per-seed parallel launches ───────────────────────────────────────────────
# We loop seeds *outside* the existing scripts so the two legs can run
# concurrently per seed. Each leg waits for both before moving to the next
# seed; that keeps GPU contention bounded and the wandb run names unique.
overall_rc=0
for seed in "${SEED_ARR[@]}"; do
    echo
    echo "──────── seed=${seed} ────────"
    log_scratch="${LOG_DIR}/compare_caco2_scratch_seed${seed}_${STAMP}.log"
    log_pretrain="${LOG_DIR}/compare_caco2_esmcv2_seed${seed}_${STAMP}.log"

    SEED="${seed}" \
    GRAPHIUM_DUMP_PREDS=1 \
    ADMET_TASKS_OVERRIDE="caco2_wang" \
        bash scripts/00_scratch_admet.sh pairmixer_12M ${GPU_SCRATCH} \
        > "${log_scratch}" 2>&1 &
    pid_s=$!

    SEED="${seed}" \
    GRAPHIUM_DUMP_PREDS=1 \
    ADMET_TASKS_OVERRIDE="caco2_wang" \
        bash scripts/00_finetune_admet.sh pairmixer_12M "${CKPT}" ${GPU_PRETRAIN} \
        > "${log_pretrain}" 2>&1 &
    pid_p=$!

    echo "  scratch    PID=${pid_s}  log=${log_scratch}"
    echo "  pretrained PID=${pid_p}  log=${log_pretrain}"

    wait ${pid_s}; rc_s=$?
    wait ${pid_p}; rc_p=$?
    printf "  → seed=%s  scratch exit=%-3d  pretrain exit=%-3d\n" "${seed}" "${rc_s}" "${rc_p}"
    if [[ ${rc_s} -ne 0 || ${rc_p} -ne 0 ]]; then
        overall_rc=1
    fi
done

# ── Aggregate ───────────────────────────────────────────────────────────────
echo
echo "== aggregating ensemble metrics =="
PYTHON=/home/shpark/.conda/envs/graphium/bin/python
# Hydra's default rundir is outputs/YYYY-MM-DD/HH-MM-SS/ — no `_ft_` substring,
# so the aggregator's default glob finds nothing. Restrict to dirs created on or
# after STAMP's date to filter out unrelated runs.
TODAY=$(date +%Y-%m-%d)
RUNS_GLOB="outputs/${TODAY}/*"

${PYTHON} scripts/aggregate_admet_ensemble.py \
    --model pairmixer_12M --pretrain scratch --tasks caco2_wang \
    --runs-glob "${RUNS_GLOB}" \
    || echo "WARN: scratch aggregation failed"
${PYTHON} scripts/aggregate_admet_ensemble.py \
    --model pairmixer_12M --pretrain "${PRETRAIN_LABEL}" --tasks caco2_wang \
    --runs-glob "${RUNS_GLOB}" \
    || echo "WARN: ${PRETRAIN_LABEL} aggregation failed"

echo
echo "== Done =="
echo "  per-seed rows  : results/experiment_results.csv"
echo "  ensemble rows  : results/experiment_results_ensemble.csv"
echo "  notebook       : notebooks/03_test_ensemble_results.ipynb"
exit ${overall_rc}
