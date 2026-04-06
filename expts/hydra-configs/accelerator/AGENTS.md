<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# expts/hydra-configs/accelerator

## Purpose
General accelerator-level configuration: which device to use and its coarse flags. This is loaded **first** in the composition order, so anything set here can be overridden by architecture/tasks/training/model configs.

## Key Files

| File | Description |
|------|-------------|
| `cpu.yaml` | Plain CPU training — used for tests and tiny debug runs |
| `gpu.yaml` | Single-GPU training via PyTorch Lightning (default for most runs) |
| `ipu.yaml` | Graphcore IPU via `poptorch` |
| `ipu_pipeline.yaml` | IPU with pipeline parallelism across multiple IPUs |

## For AI Agents

### Working In This Directory
- Keep these files **minimal**. Experiment-specific knobs (precision, dataloader workers, batch size) belong in `../training/accelerator/<dataset>_<device>.yaml`, which loads later and overrides this group.
- The Hydra override is `accelerator=gpu` (no extension).
- Adding a new device requires updating `graphium/config/_loader.py::load_accelerator` to recognize it.

### Testing Requirements
- GPU smoke: `graphium-train accelerator=gpu model=gcn tasks=toymix training=toymix`
- CPU smoke: `pytest tests/test_training.py -x` runs with `accelerator=cpu`.

## Dependencies

### Internal
- `graphium/config/_loader.py::load_accelerator`
- `../training/accelerator/` — more specific specializations

### External
- `pytorch_lightning` (Strategy selection), `poptorch` (IPU only)

<!-- MANUAL: -->
