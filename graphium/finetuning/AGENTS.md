<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/finetuning

## Purpose
Everything needed to fine-tune a pre-trained Graphium checkpoint on a new task set (e.g., ADMET benchmarks): a wrapped network that exposes a pre-trained backbone plus a new head, a Lightning callback that controls layer freezing/unfreezing, and a fingerprinting helper that extracts frozen embeddings.

## Key Files

| File | Description |
|------|-------------|
| `finetuning_architecture.py` | `FullGraphFinetuningNetwork`, `PretrainedModel`, `FinetuningHead` — builds the wrapped model and loads checkpoint weights |
| `finetuning.py` | `GraphFinetuning` Lightning callback — unfreezes layers at configurable epochs/depths |
| `fingerprinting.py` | Helpers for extracting pre-trained embeddings as a fixed feature representation |
| `utils.py` | Checkpoint path resolution, config merging helpers |
| `__init__.py` | Re-exports the classes above |

## For AI Agents

### Working In This Directory
- **Finetuning is frozen by default**: `UNFREEZE_DEPTH=0` and `EPOCH_UNFREEZE_ALL=none`, meaning only the new head trains. To unfreeze the backbone, the user must explicitly set `finetuning.unfreeze_pretrained_depth` and/or `finetuning.epoch_unfreeze_all` in the YAML override. Do not change the defaults.
- `PretrainedModel` loads the state dict from a `.ckpt` path and strips task-head weights before wrapping. If you change checkpoint key naming upstream, update the strip-prefix logic here.
- `GraphFinetuning` callback matches Lightning's `BaseFinetuning` contract — override `freeze_before_training` and `finetune_function`. Keep the epoch-triggered unfreeze deterministic and test it against `tests/test_finetuning.py`.

### Testing Requirements
- `tests/test_finetuning.py` — covers frozen-backbone and unfreeze-at-epoch paths using `config/dummy_finetuning_from_*.yaml` fixtures.

### Common Patterns
- Finetuning YAML overrides live under `expts/hydra-configs/finetuning/`. The `+finetuning=...` Hydra override is the canonical way to activate them.
- `scripts/00_finetune_admet.sh` and `scripts/00_scratch_admet.sh` are the main entry points; the former uses this module, the latter skips it entirely.

## Dependencies

### Internal
- `graphium/nn/architectures/global_architectures.py::FullGraphMultiTaskNetwork` — the backbone class
- `graphium/config/_loader.py` — instantiates the finetuning network from config
- `graphium/cli/train_finetune_test.py` — attaches the callback when finetuning is requested

### External
- `pytorch_lightning`, `torch`

<!-- MANUAL: -->
