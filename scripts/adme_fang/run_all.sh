#!/usr/bin/env bash
# Run the Polaris ADME-Fang sweep for frozen-encoder MLP baselines.
#
# Iterates ENCODERS × TASKS × SEEDS (default: 2 × 6 × 5 = 60 runs).
# Each invocation appends one row to results/<encoder>_results.csv under the
# ``benchmark == "polaris"`` filter the dashboard notebook uses.
#
# Usage:
#   bash scripts/adme_fang/run_all.sh                    # GPU 0, all defaults
#   GPU=6 bash scripts/adme_fang/run_all.sh              # pin to GPU 6
#
# Env overrides:
#   ENCODERS="mole,minimol"       # comma-sep subset of {mole, minimol, pairmixer}
#   TASKS="adme_fang_hclint,..."  # default: 6 Fang tasks
#   SEEDS="0 1 2 3 4"             # space-sep
#   GPU=0
#
# Pairmixer:
#   bash scripts/adme_fang/run_all.sh <pairmixer_ckpt|scratch>
#   prepends 'pairmixer' to ENCODERS and runs scripts/pairmixer/pairmixer_admet_eval.py.

set -euo pipefail
cd "$(dirname "$0")/../.."

eval "$(conda shell.bash hook)"
conda activate graphium-downstream 2>/dev/null || conda activate graphium

PAIRMIXER_CKPT=${1:-_}
ENCODERS=${ENCODERS:-mole,minimol}
TASKS=${TASKS:-adme_fang_hclint,adme_fang_rclint,adme_fang_perm,adme_fang_hppb,adme_fang_rppb,adme_fang_solu}
SEEDS=${SEEDS:-"0 1 2 3 4"}
GPU=${GPU:-0}

if [[ "${PAIRMIXER_CKPT}" != "_" && ",${ENCODERS}," != *",pairmixer,"* ]]; then
    ENCODERS="pairmixer,${ENCODERS}"
fi

IFS=',' read -ra ENC_LIST  <<< "${ENCODERS}"
IFS=',' read -ra TASK_LIST <<< "${TASKS}"

total=$((${#ENC_LIST[@]} * ${#TASK_LIST[@]} * $(wc -w <<< "${SEEDS}")))
echo "== ADME-Fang (Polaris) sweep: ${total} runs on GPU ${GPU}"

i=0
for enc in "${ENC_LIST[@]}"; do
    case "${enc}" in
        pairmixer) script="scripts/pairmixer/pairmixer_admet_eval.py" ;;
        *)         script="scripts/${enc}/${enc}_eval.py" ;;
    esac
    if [[ ! -f "${script}" ]]; then
        echo "  SKIP: ${script} missing"
        continue
    fi
    pairmixer_extra=()
    if [[ "${enc}" == "pairmixer" ]]; then
        case "${PAIRMIXER_CKPT}" in
            _) echo "  SKIP: pairmixer requires a .ckpt path or 'scratch'"; continue ;;
            scratch|random|none) ;;
            *) [[ -f "${PAIRMIXER_CKPT}" ]] || { echo "  SKIP: pairmixer ckpt not found: ${PAIRMIXER_CKPT}"; continue; } ;;
        esac
        pairmixer_extra=(--ckpt "${PAIRMIXER_CKPT}")
    fi
    for task in "${TASK_LIST[@]}"; do
        for seed in ${SEEDS}; do
            i=$((i + 1))
            echo
            echo "------------------------------------------------------------"
            echo "[${i}/${total}] ${enc} / ${task} / seed=${seed}"
            echo "------------------------------------------------------------"
            CUDA_VISIBLE_DEVICES=${GPU} python "${script}" \
                --benchmark polaris --task "${task}" --seed "${seed}" \
                "${pairmixer_extra[@]}" \
                || echo "  WARN: ${enc}/${task}/seed=${seed} failed; continuing"
        done
    done
done

echo
echo "Done. Rows in results/{mole,minimol}_results.csv (filter benchmark=='polaris')."
