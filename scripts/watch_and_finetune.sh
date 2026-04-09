#!/usr/bin/env bash
# Watch pre-training PIDs and launch ADMET fine-tuning in the same tmux pane.
# Usage: nohup bash scripts/watch_and_finetune.sh &

set -uo pipefail
cd "$(dirname "$0")/.."

POLL_INTERVAL=60  # seconds

# ── Jobs: PID, checkpoint dir, pretrain dataset, tmux target ────────────────
declare -A PIDS CKPT_DIRS DATASETS TMUX_TARGETS LAUNCHED

PIDS[gpu4]=2005698
CKPT_DIRS[gpu4]="models_checkpoints/toymix-lpm24/gpspp_800M/2026-04-08_11-21-36_20260408_112137"
DATASETS[gpu4]="toymix_lpm24"
TMUX_TARGETS[gpu4]="agentdeck_exp-lpm24_28682270:0.0"
LAUNCHED[gpu4]=false

PIDS[gpu5]=2261245
CKPT_DIRS[gpu5]="models_checkpoints/toymix-dti-v2/gpspp_800M/2026-04-08_14-27-43_20260408_142743"
DATASETS[gpu5]="toymix_dti_v2"
TMUX_TARGETS[gpu5]="agentdeck_exp-dti-esm2_e4dbdb88:0.0"
LAUNCHED[gpu5]=false

PIDS[gpu6]=2178235
CKPT_DIRS[gpu6]="models_checkpoints/toymix-dti-esmc-v2/gpspp_800M/2026-04-08_13-29-25_20260408_132925"
DATASETS[gpu6]="toymix_dti_esmc_v2"
TMUX_TARGETS[gpu6]="agentdeck_exp-dti-esmc_359b7fb1:0.0"
LAUNCHED[gpu6]=false

launch_finetune() {
    local key=$1
    local gpu=${key#gpu}
    local ckpt="${CKPT_DIRS[$key]}/last.ckpt"
    local dataset="${DATASETS[$key]}"
    local target="${TMUX_TARGETS[$key]}"

    if [[ ! -f "${ckpt}" ]]; then
        echo "$(date '+%F %T') [GPU ${gpu}] ERROR: ${ckpt} not found, skipping"
        return 1
    fi

    local cmd="PRETRAIN_DATASET=${dataset} bash scripts/00_finetune_admet.sh gpspp_800M ${ckpt} ${gpu}"

    echo "$(date '+%F %T') [GPU ${gpu}] Sending fine-tuning command to tmux ${target}"
    tmux send-keys -t "${target}" "${cmd}" Enter

    echo "$(date '+%F %T') [GPU ${gpu}] Fine-tuning launched in ${target}"
}

echo "$(date '+%F %T') Watching pre-training PIDs..."
for key in gpu4 gpu5 gpu6; do
    echo "  GPU ${key#gpu}: PID ${PIDS[$key]} (${DATASETS[$key]}) -> tmux ${TMUX_TARGETS[$key]}"
done

while true; do
    all_launched=true

    for key in gpu4 gpu5 gpu6; do
        if [[ "${LAUNCHED[$key]}" == "false" ]]; then
            if ! kill -0 "${PIDS[$key]}" 2>/dev/null; then
                echo "$(date '+%F %T') [GPU ${key#gpu}] Pre-training finished (${DATASETS[$key]})"
                sleep 5  # brief pause to let the process fully exit
                launch_finetune "${key}"
                LAUNCHED[$key]=true
            else
                all_launched=false
            fi
        fi
    done

    if [[ "${all_launched}" == "true" ]]; then
        echo "$(date '+%F %T') All fine-tuning jobs launched. Watcher exiting."
        break
    fi

    sleep "${POLL_INTERVAL}"
done
