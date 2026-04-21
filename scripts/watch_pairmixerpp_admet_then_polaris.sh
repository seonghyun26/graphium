#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL=${MODEL:-pairmixerpp_12M}
BASELINE_MODEL=${BASELINE_MODEL:-pairmixer_12M}
DEVICE=${DEVICE:-0}
POLL_SECS=${POLL_SECS:-300}
RESULTS_CSV=${RESULTS_CSV:-${RESULTS_DIR}/experiment_results.csv}
COMPARE_OUT=${COMPARE_OUT:-logs/${MODEL}_vs_${BASELINE_MODEL}_tdc_admet.md}
WATCH_LOG=${WATCH_LOG:-logs/${MODEL}_admet_watch.log}

mkdir -p logs

count_finished_tasks() {
python - "$RESULTS_CSV" "$MODEL" <<'PY'
import sys, pandas as pd
csv, model = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv, low_memory=False)
tags = df["wandb_tags"].astype(str)
mask = tags.str.contains(model) & tags.str.contains("scratch") & tags.str.contains("admet")
sub = df.loc[mask, ["task", "timestamp"]].copy()
if sub.empty:
    print(0)
else:
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub = sub.sort_values("timestamp").drop_duplicates("task", keep="last")
    print(sub["task"].nunique())
PY
}

list_finished_tasks() {
python - "$RESULTS_CSV" "$MODEL" <<'PY'
import sys, pandas as pd
csv, model = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv, low_memory=False)
tags = df["wandb_tags"].astype(str)
mask = tags.str.contains(model) & tags.str.contains("scratch") & tags.str.contains("admet")
sub = df.loc[mask, ["task", "timestamp"]].copy()
if sub.empty:
    print("")
else:
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub = sub.sort_values("timestamp").drop_duplicates("task", keep="last")
    print(" ".join(sub["task"].dropna().astype(str).tolist()))
PY
}

is_active() {
    pgrep -af "graphium-train model=${MODEL} .*tasks=admet|graphium-train .*model=${MODEL} .*tasks=admet" >/dev/null
}

{
    echo "[$(date '+%F %T')] Watching ${MODEL} scratch ADMET completion..."
    echo "Results CSV: ${RESULTS_CSV}"
    while true; do
        count=$(count_finished_tasks)
        finished=$(list_finished_tasks)
        echo "[$(date '+%F %T')] finished ${count}/22 tasks: ${finished}"
        if [[ "$count" -ge 22 ]]; then
            echo "[$(date '+%F %T')] All 22 ADMET tasks finished. Writing comparison..."
            /home/shpark/.conda/envs/graphium/bin/python scripts/compare_admet_runs.py \
                --csv "${RESULTS_CSV}" \
                --candidate "${MODEL}" \
                --baseline "${BASELINE_MODEL}" \
                --output "${COMPARE_OUT}"
            echo "[$(date '+%F %T')] Comparison written to ${COMPARE_OUT}"
            echo "[$(date '+%F %T')] Launching Polaris ADME-Fang scratch run on GPU ${DEVICE}..."
            bash scripts/00_scratch_polaris_admet.sh "${MODEL}" "${DEVICE}"
            break
        fi
        if ! is_active; then
            echo "[$(date '+%F %T')] No active ${MODEL} scratch ADMET process found and only ${count}/22 tasks finished."
            exit 1
        fi
        sleep "${POLL_SECS}"
    done
} | tee -a "${WATCH_LOG}"
