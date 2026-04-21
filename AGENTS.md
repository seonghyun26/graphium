<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium

## Purpose
Fork of [Graphium](https://github.com/datamol-io/graphium) extended with multi-modal pre-training (molecular properties + RxRx3 cell morphology) and ablation studies on pre-training dataset type and size. Trains GNN models (GCN, MPNN, GPS++, Pairformer, PairMixer) on molecular graph data and fine-tunes them on 22 ADMET benchmark tasks. All user-facing entry points are `graphium-train` (Hydra CLI) and the scripts under `scripts/`.

## Key Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | Detailed project architecture docs, execution flow, Hydra composition order, finetuning defaults |
| `README.md` | Upstream Graphium project README |
| `pyproject.toml` | Package metadata, dependencies, black/ruff config (line-length=110, py310/311) |
| `requirements.txt` | Runtime dependencies (non-IPU) |
| `requirements_ipu.txt` | Additional dependencies for IPU acceleration |
| `env.yml` / `enivronment.yaml` | Conda environment specs (`graphium` env) |
| `mkdocs.yml` | MkDocs site config for the `docs/` directory |
| `codecov.yml` | Coverage reporting config |
| `install_ipu.sh` / `enable_ipu.sh` | Graphcore IPU setup helpers |
| `PROTEIN_PIPELINE_HANDOFF.md` | Handoff notes for the protein (ESM) multi-modal pipeline |
| `test_results.yaml` / `val_results.yaml` | Most recent run outputs written by `graphium-train` |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `graphium/` | Main Python package — models, data, trainer, config loader (see `graphium/AGENTS.md`) |
| `expts/` | Hydra configs and experiment entry scripts (see `expts/AGENTS.md`) |
| `scripts/` | Shell wrappers for pretraining, finetuning, ablations (see `scripts/AGENTS.md`) |
| `tests/` | Pytest suite (see `tests/AGENTS.md`) |
| `notebooks/` | Jupyter notebooks for results dashboard and ablation analysis (see `notebooks/AGENTS.md`) |
| `docs/` | MkDocs documentation site (see `docs/AGENTS.md`) |
| `profiling/` | Performance profiling artefacts |
| `data/` | Raw and featurized datasets (ToyMix/LargeMix/RxRx3/ADMET) — **generated, do not document** |
| `datacache/` | Cached PyG graphs keyed by featurization hash — **generated** |
| `outputs/` | Per-run Hydra output directories — **generated** |
| `results/` | `experiment_results.csv` consumed by the dashboard notebook — **append-only** |
| `models_checkpoints/` | Saved `.ckpt` files from pretraining and ADMET finetuning — **generated** |
| `wandb/`, `logs/`, `autoresearch/` | Training artefacts — **generated** |

## For AI Agents

### Working In This Directory
- **Environment**: `mamba activate graphium`. The Python at `/home/shpark/.conda/envs/graphium/bin/python` has torch; the base env does not.
- **All commands assume `cd graphium/` first** — Hydra configs are resolved relative to this directory.
- **Do NOT modify** `data/`, `datacache/`, `outputs/`, `models_checkpoints/`, `wandb/`, `logs/`, `autoresearch/`, `results/experiment_results.csv` — these are machine-generated. `results/` is append-only via the training loop.
- **Formatting**: `black .` (line-length=110, py310/311) and `ruff check`.
- **Finetuning is frozen by default**: `UNFREEZE_DEPTH=0`, `EPOCH_UNFREEZE_ALL=none`. Only task heads train unless explicitly unfrozen.
- **GPS++ dimension gotcha**: When changing hidden dims, you must manually set `pre_nn.out_dim`, `pre_nn.hidden_dims`, `gnn.in_dim/out_dim/hidden_dims`, AND `gnn.layer_kwargs.mpnn_kwargs.in_dim/out_dim` — these are NOT auto-linked.

### Testing Requirements
```bash
pytest -m 'not ipu'                    # all non-IPU tests
pytest tests/test_architectures.py     # single file
pytest tests/test_architectures.py::test_name -x
```

### Common Patterns
- **Hydra composition order** (later overrides earlier): `accelerator → architecture → tasks → training → model → training/accelerator → training/model`.
- **Results tracking**: Every `graphium-train` run appends a row to `results/experiment_results.csv`. Notebook `notebooks/00_results_dashboard.ipynb` is the reader.
- **Scripts source `scripts/common.sh`** for shared config (W&B, ADMET task list, dimension helpers, `RESULTS_DIR`).
- **Prefer YAML config changes over CLI flag overrides in scripts** (user preference stored in project memory).

## Dependencies

### Internal
- `../data/half_life_obach.tab` (at the `prj-molrepr/` monorepo root) — shared ADMET data.
- `../esm/` — ESM protein models used for multi-modal experiments (rarely modified from here).

### External
- PyTorch, PyTorch Lightning, PyTorch Geometric — core stack
- Hydra (`hydra-core`, `omegaconf`) — config composition
- `datamol`, `rdkit` — molecular featurization
- `pytdc` — ADMET benchmark downloads
- `wandb` — experiment tracking
- `poptorch` (optional) — IPU support

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
