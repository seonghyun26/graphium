<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# notebooks

## Purpose
Jupyter notebooks used for analysis, dashboards, and pilot experiments. These are **not** tested in CI — treat them as living documents that consume `results/experiment_results.csv` and cached outputs.

## Key Files

| File | Description |
|------|-------------|
| `00_results_dashboard.ipynb` | Main dashboard — reads `../results/experiment_results.csv` and plots metrics across models, datasets, and seeds |
| `01_dataset.ipynb` | Dataset exploration (label distributions, missingness, splits) |
| `02_ablations.ipynb` | Visualizes `03_ablation_*` sweeps over dataset size / type / model |
| `03_pilot_tests.ipynb` | Scratch pad for pilot architecture experiments (PairMixer, MoE, pair-pool) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `fig/` | Exported plots used in reports and slides |
| `_archive/` | Deprecated notebooks kept for historical reference |

## For AI Agents

### Working In This Directory
- The dashboard notebook is the canonical reader of `../results/experiment_results.csv`. If you add a new column to that CSV (new metric, new config knob), update the schema comments at the top of `00_results_dashboard.ipynb`.
- **Do not clear outputs on commit** unless asked — some users rely on rendered plots in the committed notebook. If the diff is noisy, ask before running `nbstripout`.
- When consuming sweep outputs, prefer `graphium/hyper_param_search/results.py` helpers over re-implementing the aggregation logic in the notebook.

### Testing Requirements
- No automated tests. Manual verification: restart kernel and run-all on a fresh clone to confirm the notebook still executes against the current CSV schema.

### Common Patterns
- Notebooks cd to the project root (`../`) at the top so relative paths match the rest of the codebase.
- Plotting: `matplotlib` + `seaborn`, with a consistent color palette across notebooks.

## Dependencies

### Internal
- `../results/experiment_results.csv` — the primary data source
- `../graphium/hyper_param_search/results.py` — aggregation helpers
- `../outputs/` — per-run Hydra outputs (read-only)

### External
- `jupyter`, `pandas`, `matplotlib`, `seaborn`, `numpy`

<!-- MANUAL: -->
