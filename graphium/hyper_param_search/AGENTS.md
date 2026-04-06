<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/hyper_param_search

## Purpose
Small helper module for consolidating hyperparameter-sweep results. Not a full HPO engine — Hydra multirun is the actual sweep driver. This module only provides post-hoc result aggregation and introspection utilities.

## Key Files

| File | Description |
|------|-------------|
| `results.py` | Load and aggregate sweep outputs from `outputs/` + `results/experiment_results.csv` |
| `__init__.py` | Re-exports aggregation helpers |

## For AI Agents

### Working In This Directory
- This is intentionally thin. If you need Bayesian HPO or early-stopping on a sweep, configure it via Hydra's multirun launcher in `expts/hydra-configs/hparam_search/`, not by growing this module.
- The notebook `notebooks/02_ablations.ipynb` is an example consumer of `results.py`.

### Testing Requirements
- No dedicated tests. Any change should be exercised by running a toy sweep:
  ```bash
  graphium-train -m model=gcn,gin tasks=toymix
  ```

## Dependencies

### Internal
- `results/experiment_results.csv` — the CSV read by `results.py`
- `graphium/utils/fs.py` — filesystem helpers

### External
- `pandas`, `hydra-core`

<!-- MANUAL: -->
