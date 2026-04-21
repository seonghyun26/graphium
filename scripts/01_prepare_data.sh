#!/usr/bin/env bash
# Pre-process and cache molecular graphs for a dataset.
# This avoids slow featurization during training.
#
# Usage:
#   bash scripts/01_prepare_data.sh <dataset> [datacache_path]
#
# Arguments:
#   dataset        : toymix | largemix | rxrx3 | largemix_rxrx3 | toymix_rxrx3
#   datacache_path : override datacache directory (default: per-dataset)
#
# Examples:
#   bash scripts/01_prepare_data.sh largemix_rxrx3
#   bash scripts/01_prepare_data.sh toymix

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${1:? "Usage: $0 <dataset> [datacache_path]"}

# ── Dataset -> config overrides ──────────────────────────────────────────────
case "${DATASET}" in
    toymix)
        TASKS=toymix
        TRAINING=toymix
        ARCHITECTURE=toymix
        DEFAULT_CACHE=/home/shpark/prj-molrepr/datacache/neurips2023-small/
        ;;
    largemix)
        TASKS=largemix
        TRAINING=largemix
        ARCHITECTURE=largemix
        DEFAULT_CACHE=/home/shpark/prj-molrepr/datacache/neurips2023-large/
        ;;
    rxrx3)
        TASKS=rxrx3
        TRAINING=rxrx3
        ARCHITECTURE=largemix
        DEFAULT_CACHE=/home/shpark/prj-molrepr/datacache/rxrx3/
        ;;
    largemix_rxrx3)
        TASKS=largemix_rxrx3
        TRAINING=largemix
        ARCHITECTURE=largemix
        DEFAULT_CACHE=/home/shpark/prj-molrepr/datacache/neurips2023-large/
        ;;
    toymix_rxrx3)
        TASKS=toymix_rxrx3
        TRAINING=toymix_rxrx3
        ARCHITECTURE=toymix
        DEFAULT_CACHE=/home/shpark/prj-molrepr/datacache/toymix_rxrx3/
        ;;
    *)
        echo "Error: unknown dataset '${DATASET}'"
        exit 1
        ;;
esac

CACHE_PATH=${2:-${DEFAULT_CACHE}}
mkdir -p "${CACHE_PATH}"

echo "============================================================"
echo "  Preparing datacache: ${DATASET}"
echo "  Cache path: ${CACHE_PATH}"
echo "============================================================"

graphium data prepare \
    model=gpspp_800M \
    tasks=${TASKS} \
    training=${TRAINING} \
    architecture=${ARCHITECTURE} \
    accelerator=cpu \
    ++datamodule.args.processed_graph_data_path=${CACHE_PATH} \
    ++datamodule.args.featurization_n_jobs=20 \
    ++constants.seed=0

echo ""
echo "=== Done. Cache at: ${CACHE_PATH} ==="
du -sh "${CACHE_PATH}"
