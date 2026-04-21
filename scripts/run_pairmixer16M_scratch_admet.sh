#!/usr/bin/env bash
# Train PairMixer 16M from scratch on all 22 ADMET tasks.
#
# Usage:
#   bash scripts/run_pairmixer16M_scratch_admet.sh [gpu_id]
#
# Defaults:
#   gpu_id = 3

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_ID=${1:-3}

bash scripts/00_scratch_admet.sh pairmixer_16M "${GPU_ID}"
