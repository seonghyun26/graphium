#!/usr/bin/env bash
# Pilot: compare graph-aware stats pooling vs masked-mean pooling for PairMixer.
#
# Trains two pairmixer_small variants on toymix (stats vs mean pooling of pair_feat),
# then fine-tunes each on the single ADMET task caco2_wang.
#
# Models:
#   pairmixer_small       — uses pair_pool=stats (graph-aware, 6*pair_dim features)
#   pairmixer_small_mean  — uses pair_pool=mean  (legacy, 1*pair_dim features)
# Both configs are otherwise identical.
#
# Usage:
#   bash scripts/04_pilot_pair_pool.sh [gpu_id]
#
# Runs sequentially: pretrain(stats) -> pretrain(mean) -> finetune(stats) -> finetune(mean)

set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${1:-0}
DATASET=toymix
FT_TASK=caco2_wang

echo "=== Pilot: pair_pool stats vs mean (pairmixer_small on ${DATASET} -> ${FT_TASK}) ==="
echo "GPU: ${GPU}"
echo

# ── 1. Pretrain with stats pooling (default pairmixer_small config) ──────────
echo ">>> [1/4] Pretrain pairmixer_small (pair_pool=stats) on ${DATASET}"
bash scripts/00_pretrain.sh pairmixer_small ${DATASET} ${GPU}

CKPT_STATS=$(ls -t models_checkpoints/small-dataset/pairmixer_small/*/toymix_pairmixer_small_*.ckpt 2>/dev/null | head -1)
echo "Stats checkpoint: ${CKPT_STATS}"
echo

# ── 2. Pretrain with masked-mean pooling ─────────────────────────────────────
echo ">>> [2/4] Pretrain pairmixer_small_mean (pair_pool=mean) on ${DATASET}"
bash scripts/00_pretrain.sh pairmixer_small_mean ${DATASET} ${GPU}

CKPT_MEAN=$(ls -t models_checkpoints/small-dataset/pairmixer_small_mean/*/toymix_pairmixer_small_mean_*.ckpt 2>/dev/null | head -1)
echo "Mean checkpoint: ${CKPT_MEAN}"
echo

# ── 3. Fine-tune stats checkpoint on caco2_wang ──────────────────────────────
echo ">>> [3/4] Fine-tune stats ckpt on ${FT_TASK}"
ADMET_TASKS_OVERRIDE="${FT_TASK}" MODEL_TAG="pilot_pair_pool_stats" \
    bash scripts/00_finetune_admet.sh pairmixer_small "${CKPT_STATS}" ${GPU}

# ── 4. Fine-tune mean checkpoint on caco2_wang ───────────────────────────────
echo ">>> [4/4] Fine-tune mean ckpt on ${FT_TASK}"
ADMET_TASKS_OVERRIDE="${FT_TASK}" MODEL_TAG="pilot_pair_pool_mean" \
    bash scripts/00_finetune_admet.sh pairmixer_small_mean "${CKPT_MEAN}" ${GPU}

echo
echo "=== Done. Compare rows where wandb_tags contains 'pilot_pair_pool' in results/experiment_results.csv ==="
