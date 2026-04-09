#!/usr/bin/env bash
# Fine-tune dti_v2 (GPU 5) and dti_esmc_v2 (GPU 6) on ADMET.

# GPU 5: toymix_dti_v2
PRETRAIN_DATASET=toymix_dti_v2 \
    bash scripts/00_finetune_admet.sh gpspp_800M \
    models_checkpoints/toymix-dti-v2/gpspp_800M/2026-04-08_14-27-43_20260408_142743/last.ckpt 5 &

# GPU 6: toymix_dti_esmc_v2
PRETRAIN_DATASET=toymix_dti_esmc_v2 \
    bash scripts/00_finetune_admet.sh gpspp_800M \
    models_checkpoints/toymix-dti-esmc-v2/gpspp_800M/2026-04-08_13-29-25_20260408_132925/last.ckpt 6 &

wait
echo "All done."
