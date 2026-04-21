<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# scripts

## Purpose
Shell entry points for pretraining, finetuning, and ablation runs. Every script thinly wraps `graphium-train` with a handful of Hydra overrides; shared config lives in `common.sh`.

## Key Files

| File | Description |
|------|-------------|
| `common.sh` | **Shared config** — W&B settings, ADMET task list, dimension helpers, `RESULTS_DIR`, utility functions. All 0*.sh scripts source this |
| `00_pretrain.sh` | Pretrain a model (`MODEL DATASET GPU_ID`) |
| `00_finetune_admet.sh` | Finetune a pre-trained checkpoint on ADMET (`MODEL CKPT_PATH GPU_ID`), frozen backbone by default |
| `00_scratch_admet.sh` | ADMET baselines trained from scratch (`MODEL GPU_ID`) |
| `00_pretrain_finetune.sh` | Pretrain → finetune pipeline in one run |
| `00_debug.sh` | Fast sanity run (3 epochs, 10 batches) for a model+dataset combo |
| `01_prepare_data.sh` | Download / featurize / cache a dataset |
| `02_pilot_moe.sh` | Mixture-of-experts pilot sweep |
| `03_ablation_dataset_size.sh` | Ablation: vary pretraining dataset size |
| `03_ablation_dataset_type.sh` | Ablation: vary pretraining dataset composition |
| `03_ablation_full_matrix.sh` | Full (dataset_type × dataset_size × model) ablation matrix |
| `04_pilot_pair_pool.sh` | PairMixer pair-pool pilot |
| `bench_featurization.py` | Micro-benchmark for the SMILES featurizer |
| `convert_yml.py` | YAML conversion helper |
| `pip_install.sh` | Project `pip install` helper |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `dti/` | DTI (drug-target interaction) finetuning scripts |
| `test/` | Manual test runner scripts |
| `data/` | One-off data prep helpers |
| `_archive/` | Deprecated scripts kept for reference |

## For AI Agents

### Working In This Directory
- **Always source `common.sh`** in new scripts — it sets `RESULTS_DIR`, W&B env vars, and the ADMET task list.
- **Prefer YAML config changes over CLI flag overrides** (project memory). If a script needs a permanent override, add it to the relevant config under `expts/hydra-configs/` instead of littering the script with `++overrides`.
- **Ablation scripts use 5 representative ADMET tasks** (one per category), not the full 22-task sweep (project memory). Do not expand to the full list without an explicit request.
- Scripts take positional arguments `MODEL`, `DATASET`, `GPU_ID` in that order where applicable.
- GPU selection is `CUDA_VISIBLE_DEVICES=$GPU_ID` — 8× RTX PRO 6000 Blackwell Max-Q (96GB each) are available.

### Testing Requirements
- Run `bash scripts/00_debug.sh gpspp toymix 0` to smoke-test changes to shared config in `common.sh`.

### Common Patterns
- All scripts `cd` to the project root at the top so Hydra can find `expts/hydra-configs/main.yaml`.
- Output logs go to `../logs/` and checkpoints to `../models_checkpoints/<dataset>/<model>/<timestamp>/`.
- `run_training_finetuning_testing` appends to `../results/experiment_results.csv` automatically — scripts should not write to that CSV directly.

## Dependencies

### Internal
- `expts/hydra-configs/` — resolves all `model=`, `tasks=`, `training=`, `accelerator=` overrides
- `graphium-train` CLI (from `graphium/cli/train_finetune_test.py`)

### External
- `bash`, `conda`/`mamba` (`graphium` env), `nvidia-smi`, optional `wandb` login

<!-- MANUAL: -->
