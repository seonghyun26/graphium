#!/usr/bin/env bash
# Setup conda environment for ESM-C embedding extraction.
#
# ESM-C (EvolutionaryScale) requires Python >=3.12,<3.13.
# This env is used ONLY for Stage 2 (02_extract_embeddings_esmc.py).
# Stages 3-5 run in the main `graphium` env (Python 3.9).
#
# Usage:
#     bash setup_esmc_env.sh

set -euo pipefail

ENV_NAME="esmc"

echo "Creating conda env '${ENV_NAME}' with Python 3.12 ..."
mamba create -n "${ENV_NAME}" python=3.12 -y

echo "Installing PyTorch with CUDA 12.8 support ..."
mamba run -n "${ENV_NAME}" pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "Installing ESM-C ..."
mamba run -n "${ENV_NAME}" pip install esm

# Optional: flash-attn for faster inference (requires CUDA toolkit)
echo "Attempting to install flash-attn (optional, may fail without CUDA toolkit) ..."
mamba run -n "${ENV_NAME}" pip install flash-attn --no-build-isolation 2>/dev/null \
    && echo "  flash-attn installed." \
    || echo "  flash-attn install failed (optional, continuing without it)."

echo ""
echo "Done. Verify with:"
echo "  conda run -n ${ENV_NAME} python -c \"from esm.models.esmc import ESMC; print('ESM-C OK')\""
