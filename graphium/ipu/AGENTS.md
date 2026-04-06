<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/ipu

## Purpose
Graphcore IPU (Poplar/PopTorch) acceleration support: IPU-specific dataloader, loss/metric implementations that avoid host-device sync, and a Lightning wrapper that substitutes PopTorch for `torch`. This subpackage is **optional**: unit tests under `tests/test_ipu_*.py` are skipped unless `poptorch` is installed.

## Key Files

| File | Description |
|------|-------------|
| `ipu_dataloader.py` | IPU-compatible dataloader with packing and padding |
| `ipu_losses.py` | Device-friendly reimplementations of standard losses |
| `ipu_metrics.py` | Device-friendly reimplementations of standard metrics |
| `ipu_simple_lightning.py` | PyTorch Lightning + PopTorch glue |
| `ipu_wrapper.py` | Wraps a Graphium model into an IPU-compiled module |
| `ipu_utils.py` | IPU option helpers, memory diagnostics |
| `to_dense_batch.py` | Dense-batch conversion used by attention-style layers on IPU |
| `__init__.py` | Re-exports IPU helpers |
| `README.md` | Upstream folder guide |

## For AI Agents

### Working In This Directory
- **Do not import `poptorch` at module import time** — guard imports so the rest of the package stays importable without IPU.
- Losses and metrics here intentionally duplicate `graphium/trainer/` equivalents. Keep the mathematical behavior identical; the only reason to touch these is to add IPU-specific ops (e.g., replacing dynamic ops with static equivalents).
- IPU tests require Graphcore SDK + `poptorch` and are marked with `@pytest.mark.ipu`. Run non-IPU tests with `pytest -m 'not ipu'`.
- `scripts/install_ipu.sh` and `enable_ipu.sh` set up the SDK on the dev machine.

### Testing Requirements
- `tests/test_ipu_*.py` — each covers a slice of this subpackage (dataloader, losses, metrics, options, poptorch, to_dense_batch).

## Dependencies

### Internal
- `graphium/trainer/losses.py`, `graphium/trainer/metrics.py` — non-IPU reference implementations

### External
- `poptorch` (Graphcore), `pytorch_lightning`, `torch`

<!-- MANUAL: -->
