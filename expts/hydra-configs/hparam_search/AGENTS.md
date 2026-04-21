<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-06 | Updated: 2026-04-06 -->

# expts/hydra-configs/hparam_search

## Purpose
Hydra multirun / sweeper configurations. Currently only an Optuna sweeper definition; most sweeps are still driven via Hydra's bash-level multirun (`graphium-train -m model=a,b,c`).

## Key Files

| File | Description |
|------|-------------|
| `optuna.yaml` | Optuna sweeper config — search space, algorithm, number of trials |

## For AI Agents

### Working In This Directory
- Activate with `+hparam_search=optuna -m`.
- Post-run aggregation is handled by `graphium/hyper_param_search/results.py` + `notebooks/02_ablations.ipynb`.
- Keep search-space definitions minimal and dataset-family agnostic here — put dataset-specific sweeps in dedicated shell scripts under `scripts/03_ablation_*.sh`.

## Dependencies

### Internal
- `graphium/hyper_param_search/results.py` — result aggregation

### External
- `hydra-optuna-sweeper`, `optuna`

<!-- MANUAL: -->
