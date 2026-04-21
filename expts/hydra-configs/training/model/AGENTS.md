<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-06 | Updated: 2026-04-06 -->

# expts/hydra-configs/training/model

## Purpose
`(dataset, model)`-specific specializations — the **last** group loaded in the Hydra composition chain, which means every key set here has the highest priority and will override any earlier group. Primary use: run name, checkpoint directory, batch size, LR, and per-combination scheduler tuning.

## Key Files
Naming: `<dataset>_<model>.yaml`. Current file count is ~90 and grows with each new (dataset, model) combination.

Representative examples:

| File | Description |
|------|-------------|
| `toymix_gpspp.yaml` / `toymix_gpspp_800M.yaml` | GPS++ on ToyMix, two scales |
| `largemix_gpspp_800M.yaml` / `largemix_gpspp_deep.yaml` | GPS++ on LargeMix |
| `toymix_pairformer.yaml` / `toymix_pairformer_{small,medium,large,boltz}.yaml` | Pairformer scale variants on ToyMix |
| `toymix_pairmixer_{small,medium,boltz}.yaml`, `toymix_dti_pairmixer_*.yaml` | PairMixer sweeps |
| `largemix_{gcn,gin,gine,gated_gcn,mpnn,pairformer}.yaml` | Non-GPS++ models on LargeMix |
| `rxrx3_{gcn,mpnn,gpspp*,pairformer,pairformer_boltz}.yaml` | Models on RxRx3 |
| `dti_{gcn,mpnn,gpspp,gpspp_800M,pairformer,pairformer_boltz}.yaml` | Models on DTI |
| `toymix_bbbc047_*.yaml`, `toymix_rxrx3_*.yaml`, `toymix_dti_*.yaml` | Multi-modal combinations |
| `largemix_dti_*_gpspp_800M.yaml`, `largemix_rxrx3_*_gpspp_800M.yaml` | Large multi-modal runs |

See `ls` output for the full list — do not try to enumerate every file here; the directory is the source of truth.

## For AI Agents

### Working In This Directory
- **This is the highest-priority group** in the composition chain. If a config change needs to override a scheduler, batch size, or trainer flag set anywhere else, this is the right place.
- Typical content:
  ```yaml
  # @package _global_
  constants:
    name: <run_name>
  trainer:
    model_checkpoint:
      dirpath: models_checkpoints/<subdir>/${now:%Y-%m-%d_%H-%M-%S}/
    trainer:
      max_epochs: ...
  predictor:
    optim_kwargs:
      lr: ...
  ```
- **Every new `(dataset, model)` combo needs a file here**. Without it, Hydra will resolve missing values from the parent group and the run name / checkpoint directory will default to a generic value — rarely what you want.
- **Prefer YAML config changes here over CLI `++` overrides in scripts** (project memory). Scripts should stay thin.
- **GPS++ dimension gotcha** applies to any GPS++ file here that overrides hidden dims — audit `pre_nn.out_dim`, `pre_nn.hidden_dims`, `gnn.in_dim/out_dim/hidden_dims`, `gnn.layer_kwargs.mpnn_kwargs.in_dim/out_dim`.
- **Pooling mode switches**: keep head `hidden_dims`/`out_dim` unchanged — only touch `level_in_dim` wiring (project memory).

### Testing Requirements
- `bash ../../../../scripts/00_debug.sh <model> <dataset> 0` is the fastest end-to-end validation.

## Dependencies

### Internal
- Composes with `../<dataset>.yaml`, `../accelerator/<dataset>_<device>.yaml`, `../../model/<model>.yaml`, `../../architecture/<arch>.yaml`, `../../tasks/<dataset>.yaml`.

<!-- MANUAL: -->
