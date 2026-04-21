#!/usr/bin/env bash
# Build a conda env for running downstream evals (gram_dti / tdc_dti_regression).
#
# Installs:
#   - GPU-enabled PyTorch (cu121 wheels)
#   - fair-esm (provides the esm-extract CLI used by ESM-2 extraction)
#   - autogluon.tabular (the TabularPredictor head, matching GRAM-DTI's repo)
#   - graphium (editable install, required for the PairMixer encoder)
#   - minimol (MiniMol encoder; optional — comment out if unused)
#
# MolE isn't on PyPI; its checkpoint + repo are fetched on first use via
# ``scripts/mole/download_mole_baseline.sh``.
#
# Usage:
#   bash scripts/setup_downstream_env.sh [env_name]   # default: graphium-downstream
#
# Activate:
#   conda activate graphium-downstream

set -euo pipefail

ENV_NAME=${1:-graphium-downstream}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "== env ${ENV_NAME} already exists — installing into it =="
else
    echo "== creating env ${ENV_NAME} (Python 3.10) =="
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"

echo "== installing GPU PyTorch (cu121) =="
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

echo "== installing fair-esm (ESM-2 extractor) =="
pip install fair-esm

echo "== installing autogluon.tabular =="
# autogluon.tabular[all] pulls lightgbm / xgboost / catboost; skip [all] if you
# only need the default LightGBM+ExtraTrees presets.
pip install "autogluon.tabular[all]"

echo "== installing core downstream deps =="
pip install scikit-learn pandas numpy scipy tqdm pyarrow joblib PyTDC rdkit

echo "== installing minimol (MiniMol encoder) =="
pip install minimol || echo "  WARN: minimol install failed; --encoder minimol will be unavailable."

echo "== installing graphium (editable, no deps) =="
pip install --no-deps -e "${REPO_ROOT}"

echo
echo "== verifying =="
python - <<'PY'
import importlib, sys
checks = [
    ("torch",                 lambda m: (m.__version__, m.cuda.is_available())),
    ("esm",                   lambda m: m.__version__),
    ("autogluon.tabular",     lambda m: m.__version__),
    ("sklearn",               lambda m: m.__version__),
    ("pandas",                lambda m: m.__version__),
    ("scipy",                 lambda m: m.__version__),
    ("tqdm",                  lambda m: m.__version__),
]
ok = True
for name, fn in checks:
    try:
        mod = importlib.import_module(name)
        print(f"  OK  {name:20s} {fn(mod)}")
    except Exception as exc:
        print(f"  FAIL {name:20s} {exc}")
        ok = False
# Optional: minimol
try:
    import minimol
    print(f"  OK  minimol             (optional)")
except Exception:
    print(f"  SKIP minimol             (optional — install failed or not present)")

# GPU sanity check
import torch
if torch.cuda.is_available():
    print(f"  CUDA devices: {torch.cuda.device_count()}  ({torch.cuda.get_device_name(0)})")
else:
    print(f"  WARN: torch.cuda.is_available() is False — extraction will fall back to CPU")
sys.exit(0 if ok else 1)
PY

echo
echo "== env ${ENV_NAME} ready =="
echo "Activate with:  conda activate ${ENV_NAME}"
echo
echo "Smoke test (no data needed):"
echo "  python -m downstream.tasks.gram_dti.eval --help"
echo "  esm-extract --help"
