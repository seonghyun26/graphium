#!/usr/bin/env bash
# Evaluate Minimol embeddings on TDC ADMET (22 tasks) and Polaris ADME-Fang (6 tasks).
#
# Usage:
#   bash scripts/minimol/run_minimol_admet.sh [gpu_id]
#
# Env vars:
#   BENCHMARK  : tdc | polaris | both (default: both)
#   HEAD       : linear | mlp (default: mlp)
#   SEEDS      : space-separated seeds (default: "0 1 2")
#   TASKS_OVERRIDE : override the task list (space-separated)
#
# Writes per-run rows to results/minimol_results.csv.

set -euo pipefail
cd "$(dirname "$0")/../.."

eval "$(conda shell.bash hook)"
conda activate graphium

DEVICE=${1:-7}
BENCHMARK=${BENCHMARK:-both}
HEAD=${HEAD:-mlp}
SEEDS=${SEEDS:-"0 1 2"}

TDC_TASKS=(
    caco2_wang hia_hou pgp_broccatelli bioavailability_ma
    lipophilicity_astrazeneca solubility_aqsoldb bbb_martins ppbr_az vdss_lombardo
    cyp2d6_veith cyp3a4_veith cyp2c9_veith
    cyp2c9_substrate_carbonmangels cyp2d6_substrate_carbonmangels cyp3a4_substrate_carbonmangels
    half_life_obach clearance_hepatocyte_az clearance_microsome_az
    ld50_zhu herg ames dili
)
POLARIS_TASKS=(
    adme_fang_hclint adme_fang_rclint adme_fang_perm
    adme_fang_hppb adme_fang_rppb adme_fang_solu
)

if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
    read -ra OVERRIDE_TASKS <<< "${TASKS_OVERRIDE}"
    TDC_TASKS=("${OVERRIDE_TASKS[@]}")
    POLARIS_TASKS=("${OVERRIDE_TASKS[@]}")
fi

FAILED=()
run_one() {
    local bench=$1 task=$2 seed=$3
    echo ""
    echo "=== [${bench}] ${task} seed=${seed} ==="
    CUDA_VISIBLE_DEVICES=${DEVICE} python scripts/minimol/minimol_eval.py \
        --benchmark "${bench}" --task "${task}" --seed "${seed}" --head "${HEAD}" \
        || FAILED+=("${bench}:${task}:seed${seed}")
}

if [[ "${BENCHMARK}" == "tdc" || "${BENCHMARK}" == "both" ]]; then
    for task in "${TDC_TASKS[@]}"; do
        for seed in ${SEEDS}; do
            run_one tdc "${task}" "${seed}"
        done
    done
fi

if [[ "${BENCHMARK}" == "polaris" || "${BENCHMARK}" == "both" ]]; then
    for task in "${POLARIS_TASKS[@]}"; do
        for seed in ${SEEDS}; do
            run_one polaris "${task}" "${seed}"
        done
    done
fi

echo ""
echo "================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "=== ALL RUNS SUCCEEDED ==="
else
    echo "=== FAILED (${#FAILED[@]}): ${FAILED[*]} ==="
fi
echo "Results: results/minimol_results.csv"
