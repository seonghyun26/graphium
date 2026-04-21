[![PyPI](https://img.shields.io/pypi/v/graphium)](https://pypi.org/project/graphium/)
[![Conda](https://img.shields.io/conda/v/conda-forge/graphium?label=conda&color=success)](https://anaconda.org/conda-forge/graphium)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/datamol-io/graphium/blob/main/LICENSE)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

A deep learning framework for molecular graph representation learning.
Built on [Graphium](https://github.com/datamol-io/graphium), this fork extends the platform with **multi-modal pre-training** (molecular properties + cell morphology) and a systematic **ablation study** on pre-training dataset type and size.

## Key capabilities

- **Multi-modal pre-training** across molecular (LargeMix) and biological (RxRx3 cell morphology) domains
- **4 GNN architectures**: GCN, MPNN, GPS++, Pairformer
- **4 pre-training datasets** spanning quantum, bioassay, gene expression, and cell imaging data
- **22-task ADMET benchmark** for downstream evaluation via TDC
- **Dataset ablation scripts** for studying the effect of pre-training data type and size on transfer learning

## Datasets

### Pre-training datasets

| Dataset | Tasks | Labels | Domain |
|---------|-------|--------|--------|
| **ToyMix** | QM9 (19), Tox21 (12), ZINC (3) | ~34 | Quantum mechanics, toxicity, molecular properties |
| **LargeMix** | L1000-VCAP (2934), L1000-MCF7 (2934), PCBA (1328), PCQM4M-G25 (25), PCQM4M-N4 (4) | ~7225 | Gene expression, bioassay screening, quantum properties |
| **RxRx3** | Cell morphology embeddings (384) | 384 | Cell imaging phenotypes from compound perturbations |
| **LargeMix+RxRx3** | All of the above combined | ~7609 | Multi-modal: molecular + biological |

### Downstream benchmark

**ADMET** (22 tasks from TDC): absorption (9), metabolism (6), excretion (3), toxicity (4). Each task is evaluated independently after fine-tuning from a pre-trained checkpoint.

## Model architectures

| Model | Type | Key feature |
|-------|------|-------------|
| **GCN** | Message passing | Simple, scalable baseline |
| **MPNN** | Message passing | Explicit edge embeddings |
| **GPS++** | Graph transformer | MPNN + full self-attention + positional encodings |
| **Pairformer** | Dual-track transformer | Separate node and pairwise interaction tracks (adapted from [Boltz](https://github.com/jwohlwend/boltz)) |

## Installation

```bash
# Create environment
mamba env create -f env.yml -n graphium

# For a specific CUDA version:
# CONDA_OVERRIDE_CUDA=11.8 mamba env create -f env.yml -n graphium

# Install in dev mode
mamba activate graphium
pip install --no-deps -e .
```

## Scripts

All experiment scripts live in `scripts/` and share configuration via `common.sh`.

| Script | Purpose |
|--------|---------|
| `common.sh` | Shared configuration (W&B, task lists, dimension helpers) |
| `00_pretrain.sh` | Pre-train any model on any dataset |
| `00_finetune_admet.sh` | Fine-tune a checkpoint on all 22 ADMET tasks |
| `00_scratch_admet.sh` | Train from scratch on ADMET (no pre-training baseline) |
| `00_pretrain_finetune.sh` | End-to-end: pre-train then fine-tune |
| `00_debug.sh` | Quick sanity check (3 epochs, 10 batches) |
| `01_prepare_data.sh` | Pre-cache featurized molecular graphs |
| `02_pilot_moe.sh` | MoE pilot: pre-train and/or fine-tune with Mixture of Experts |
| `03_ablation_dataset_type.sh` | Ablation study: pre-training dataset type |
| `03_ablation_dataset_size.sh` | Ablation study: pre-training dataset size |
| `03_ablation_full_matrix.sh` | Full ablation matrix (type x size) |

Old scripts are preserved in `scripts/_archive/`.

### Quick start

```bash
# Pre-train GPS++ on LargeMix
bash scripts/00_pretrain.sh gpspp largemix 0

# Pre-train GCN on the combined multi-modal dataset
bash scripts/00_pretrain.sh gcn largemix_rxrx3 0

# Fine-tune on ADMET from a checkpoint
bash scripts/00_finetune_admet.sh gpspp ./checkpoints/last.ckpt 0

# Train from scratch (baseline)
bash scripts/00_scratch_admet.sh gpspp 0

# Debug run (tiny model, 3 epochs)
bash scripts/00_debug.sh gpspp toymix 0
```

### Environment variable overrides

All scripts accept environment variables for customization:

```bash
# Custom dimensions and depth
DIM=1024 GNN_DEPTH=16 bash scripts/00_pretrain.sh gpspp largemix 0

# Fine-tuning keeps the backbone FROZEN by default (only trains the head).
# To unfreeze layers, explicitly set:
UNFREEZE_DEPTH=4 EPOCH_UNFREEZE_ALL=40 bash scripts/00_finetune_admet.sh gcn ./ckpt.ckpt 0

# Subsample pre-training data (for size ablation)
SAMPLE_SIZE=0.1 bash scripts/00_pretrain.sh gpspp largemix 0
```

## Ablation study

The ablation study isolates the effect of pre-training data on downstream ADMET performance.

### Dataset type ablation

Fixes model architecture, size, and finetuning protocol. Varies only the pre-training dataset.

```bash
# Run with GPS++ (default)
bash scripts/03_ablation_dataset_type.sh 0

# Run with GCN
MODEL=gcn DIM=5120 bash scripts/03_ablation_dataset_type.sh 0
```

**Conditions**: scratch (no pre-training), ToyMix, LargeMix, RxRx3, LargeMix+RxRx3

### Dataset size ablation

Fixes everything including dataset type. Varies the fraction of pre-training data.

```bash
# Default: subsample LargeMix at {1%, 5%, 10%, 25%, 50%, 100%}
bash scripts/03_ablation_dataset_size.sh 0

# Custom fractions on RxRx3
DATASET=rxrx3 FRACTIONS="0.1 0.5 1.0" bash scripts/03_ablation_dataset_size.sh 0
```

### Full matrix

Crosses dataset type with dataset size for a complete picture.

```bash
bash scripts/03_ablation_full_matrix.sh 0
```

Produces: 4 dataset types x 4 fractions + 1 scratch baseline = 17 pre-training conditions, each fine-tuned on all 22 ADMET tasks.

### W&B tracking

All ablation runs are tagged for filtering:

- `ablation` + `dataset_type` for type ablation
- `ablation` + `dataset_size` + `frac_X.XX` for size ablation
- `ablation` + `full_matrix` for the combined study

## Pipeline

```
Pre-training dataset                 Downstream evaluation
(ToyMix / LargeMix / RxRx3 / ...)   (22 ADMET tasks)
        |                                    ^
        v                                    |
  [00_pretrain.sh]                  [00_finetune_admet.sh]
  Multi-task GNN training    --->   Frozen backbone + task head
  on shared backbone                (only head is trained by default)
        |                                    |
        v                                    v
                    results/experiment_results.csv
                    (centralized metrics, auto-appended per run)
                                             |
                                             v
                    notebooks/00_results_dashboard.ipynb
                    (visualization: heatmaps, scaling curves, bar charts)
```

Fine-tuning keeps the pre-trained backbone **frozen by default** — only the task head is trained. This isolates the quality of the learned representations from the fine-tuning dynamics.

For the ablation study, this pipeline is run for each (dataset type, dataset size) combination, plus a scratch baseline that skips pre-training entirely.

## Results

Every `graphium-train` run automatically appends a row to `results/experiment_results.csv` with:
- Run metadata: model, task, seed, checkpoint path, finetuning config
- All test metrics from the run

Open `notebooks/00_results_dashboard.ipynb` to visualize:
- Per-task metrics table across pre-training datasets
- Dataset type heatmap (task x dataset)
- Dataset size scaling curves
- Aggregate normalized performance bar charts

## Hydra configuration

Experiments are configured with [Hydra](https://hydra.cc/). Config files are in `expts/hydra-configs/`:

```
hydra-configs/
  model/          gcn, mpnn, gpspp, gpspp_deep, pairformer, ...
  tasks/          toymix, largemix, rxrx3, largemix_rxrx3, admet, ...
  training/       toymix, largemix, rxrx3 (LR, warmup, scheduler)
  architecture/   toymix, largemix (pre-NN, PE encoders, graph output)
  finetuning/     admet, admet_largemix, admet_gpspp
```

Any parameter can be overridden from the command line:

```bash
graphium-train model=gpspp tasks=largemix training=largemix architecture=largemix \
    ++architecture.gnn.depth=16 \
    ++datamodule.args.batch_size_training=256
```

## Data preparation

Featurization (SMILES to PyG graphs) runs automatically on first training. For large datasets, prepare and cache in advance:

```bash
# Cache the featurized graphs
graphium data prepare ++datamodule.args.processed_graph_data_path=./datacache/largemix

# Train using the cache
graphium-train [...] datamodule.args.processed_graph_data_path=./datacache/largemix
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `00_results_dashboard` | **Main visualization**: reads from `results/` and plots ablation results |
| `01_dataset` | Dataset overview, embedding stats, coverage analysis, filtered dataset creation |
| `02_ablations` | MoE ablation, Pairformer size scaling, Pairformer vs Pairmixer |
| `03_pilot_tests` | Model inference demo, ADMET fine-tuning tutorial, embedding exploration |

Dev/scratch notebooks are in `notebooks/_archive/`.

## Project structure

```
graphium/
  data/
    rxrx3/                    RxRx3 cell morphology embeddings
    dti/                      Drug-target interaction data
  expts/
    hydra-configs/            Hydra YAML configurations
  graphium/                   Core library (models, data, training)
  notebooks/
    00_results_dashboard.ipynb  <-- main results visualization
    01_dataset.ipynb            Dataset overview and coverage analysis
    02_ablations.ipynb          Ablation studies
    03_pilot_tests.ipynb        Pilot tests and exploration
    fig/                      Publication figure notebooks
    _archive/                 Dev/scratch notebooks
  results/
    experiment_results.csv    Centralized metrics (auto-populated)
  scripts/
    common.sh                 Shared config
    00_*.sh                   Core training (pretrain, finetune, scratch, debug)
    01_prepare_data.sh        Data caching
    02_pilot_moe.sh           MoE pilot experiments
    03_ablation_*.sh          Ablation study scripts
    _archive/                 Old/completed scripts
  datacache/                  Cached featurized graphs
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgments

Built on [Graphium](https://github.com/datamol-io/graphium) by Valence Labs. Pairformer architecture adapted from [Boltz](https://github.com/jwohlwend/boltz). RxRx3 cell morphology data from [Recursion](https://www.rxrx.ai/).
