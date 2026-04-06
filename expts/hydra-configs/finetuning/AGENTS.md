<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-06 | Updated: 2026-04-06 -->

# expts/hydra-configs/finetuning

## Purpose
Finetuning overlays. These configs are activated with the Hydra override `+finetuning=<stem>` and layer finetuning-specific options on top of a pretraining config: pre-trained checkpoint path, new task heads, unfreeze schedule, finetuning optimizer settings.

## Key Files

| File | Description |
|------|-------------|
| `admet.yaml` | ADMET-22 finetuning base — used by the default scratch-vs-finetune comparisons |
| `admet_baseline.yaml` | No-pretrain ADMET baseline config |
| `admet_gpspp.yaml` | GPS++ → ADMET finetuning |
| `admet_largemix.yaml`, `admet_largemix_dti.yaml`, `admet_largemix_dti_filtered.yaml` | LargeMix pre-trained → ADMET |
| `admet_largemix_rxrx3.yaml`, `admet_largemix_rxrx3_dti.yaml` | Multi-modal LargeMix+RxRx3 → ADMET |
| `admet_rxrx3.yaml`, `admet_rxrx3_dti.yaml` | RxRx3 pre-trained → ADMET |
| `admet_toymix_bbbc047.yaml`, `admet_toymix_dti.yaml`, `admet_toymix_dti_filtered.yaml`, `admet_toymix_dti_10k_filtered.yaml`, `admet_toymix_rxrx3.yaml`, `admet_toymix_rxrx3_dti.yaml` | ToyMix multi-modal variants → ADMET |
| `admet_dti.yaml` | DTI-pretrained → ADMET |

## For AI Agents

### Working In This Directory
- **Finetuning is frozen by default** — `UNFREEZE_DEPTH=0`, `EPOCH_UNFREEZE_ALL=none`. Only the new task head trains. To unfreeze the backbone, set `finetuning.unfreeze_pretrained_depth` and/or `finetuning.epoch_unfreeze_all` explicitly in the YAML. Do not change the defaults globally.
- The Hydra override is `+finetuning=<stem>` (note the leading `+`).
- Finetuning YAMLs specify `finetuning.pretrained_model_path` pointing into `models_checkpoints/` — make sure the checkpoint exists before triggering the run.
- **Ablation scripts use 5 representative ADMET tasks** (one per category), not the full 22 (project memory). When adding a new finetuning override for ablations, stick to the 5-task subset.
- New finetuning YAMLs should load a sibling `tasks/admet.yaml` (or the right target) and only override: `finetuning.*`, `constants.name`, `trainer.model_checkpoint.dirpath`, `predictor.optim_kwargs.lr`.

### Testing Requirements
- `tests/test_finetuning.py` exercises frozen and unfrozen paths against dummy YAMLs in `graphium/config/`.
- Manual smoke: `bash ../../../scripts/00_finetune_admet.sh gpspp ./ckpt 0` against a tiny ckpt.

## Dependencies

### Internal
- `graphium/finetuning/finetuning_architecture.py::FullGraphFinetuningNetwork`
- `graphium/finetuning/finetuning.py::GraphFinetuning` (Lightning callback)
- `graphium/cli/train_finetune_test.py` — attaches the callback when `+finetuning=...` is present

<!-- MANUAL: -->
