#!/usr/bin/env bash
# Orchestrate the stabilize sweep on a single GPU:
#   Pair 1 (baseline, gradclip) concurrently -> wait -> Pair 2 (accum, bundle) concurrently.
#   If a pair's concurrent launch triggers OOM, the pair is retried serially.
#
# Usage:
#   bash scripts/pairmixer_stabilize_orchestrate.sh <gpu>

set -uo pipefail
cd "$(dirname "$0")/.."

source /home/shpark/miniforge3/etc/profile.d/conda.sh
conda activate graphium

GPU=${1:?"Usage: $0 <gpu>"}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/stabilize/orchestrator_${STAMP}"
mkdir -p "${LOG_DIR}"

echo "[$(date)] orchestrator start (gpu=${GPU}, logs=${LOG_DIR}, mode=serial)"

VARIANTS=(baseline gradclip accum bundle)

for v in "${VARIANTS[@]}"; do
    echo "[$(date)] starting variant=${v}"
    bash scripts/pairmixer_stabilize_sweep.sh "${v}" "${GPU}" \
        > "${LOG_DIR}/${v}.log" 2>&1
    RC=$?
    echo "[$(date)] variant=${v} finished (rc=${RC})"
    if [[ ${RC} -ne 0 ]]; then
        echo "[$(date)] FAILED: ${v} — see ${LOG_DIR}/${v}.log"
    fi
done

echo "[$(date)] orchestrator complete"
echo "logs: ${LOG_DIR}"
