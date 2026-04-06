<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/utils

## Purpose
Cross-cutting helpers: class registries, filesystem helpers, hashing, mup scaling, custom LR schedulers, tensor helpers, packing, safe-run context manager, moving averages, and command-line glue.

## Key Files

| File | Description |
|------|-------------|
| `spaces.py` | **Central class registry** mapping string names to classes for layers, encoders, losses, metrics, schedulers, residual connections, etc. — used by every config loader |
| `fs.py` | Filesystem abstraction over local/s3/gs paths via `fsspec` |
| `hashing.py` | Stable hashes for config-to-cache-key mapping |
| `mup.py` | Mup (maximal update parameterization) helpers for width scaling |
| `custom_lr.py` | `WarmUpLinearLR` and friends — custom LR schedulers |
| `tensor.py` | Generic torch tensor helpers (masked mean/max, nan handling) |
| `packing.py` | Graph packing for efficient IPU batches |
| `safe_run.py` | Context manager that captures exceptions without killing the run |
| `arg_checker.py` | Runtime assertions on function arguments |
| `command_line_utils.py` | Pretty-printers for CLI output |
| `moving_average_tracker.py` | EMA tracker used by the predictor summaries |
| `decorators.py` | Common decorators (e.g., memoize, deprecation warning) |
| `__init__.py` | Re-exports helpers |
| `README.md` | Upstream folder guide |

## For AI Agents

### Working In This Directory
- **`spaces.py` is the single source of truth** for any class that is selectable from YAML. When adding a new layer/encoder/loss/metric/scheduler/residual type, register it here — failing to do so means the config loader cannot instantiate it.
- `fs.py` is preferred over bare `open()` whenever paths might be remote.
- `custom_lr.WarmUpLinearLR` is what most configs reference for pretraining; do not silently change its warmup math.
- `hashing.py` is used to derive featurization cache paths — any change invalidates `../../datacache/`.

### Testing Requirements
- `tests/test_utils.py`, `tests/test_mup.py`, `tests/test_packing.py`, `tests/test_data_utils.py`.

### Common Patterns
- Utility functions are written to be stateless and side-effect-free so they can be used from both CPU and IPU code paths.

## Dependencies

### Internal
- Imported by nearly every other module — this is the leaf-level utility layer.

### External
- `torch`, `fsspec`, `numpy`, `mup`

<!-- MANUAL: -->
