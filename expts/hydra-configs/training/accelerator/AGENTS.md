<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-06 | Updated: 2026-04-06 -->

# expts/hydra-configs/training/accelerator

## Purpose
Device-specific training overrides per dataset family. Naming convention: `<dataset>_<device>.yaml` (e.g., `toymix_gpu.yaml`, `largemix_ipu.yaml`). These set precision, dataloader workers, and sometimes the initial batch size.

## Key Files

| File | Description |
|------|-------------|
| `toymix_cpu.yaml`, `toymix_gpu.yaml`, `toymix_ipu.yaml` | ToyMix device specializations |
| `largemix_cpu.yaml`, `largemix_gpu.yaml`, `largemix_ipu.yaml` | LargeMix device specializations |
| `pcqm4m_gpu.yaml`, `pcqm4m_ipu.yaml` | PCQM4M |
| `rxrx3_gpu.yaml`, `rxrx3_dti_gpu.yaml` | RxRx3 |
| `dti_gpu.yaml`, `dti_filtered_gpu.yaml`, `dti_10k_gpu.yaml`, `dti_10k_filtered_gpu.yaml` | DTI variants |
| `largemix_dti_gpu.yaml`, `largemix_dti_filtered_gpu.yaml`, `largemix_rxrx3_dti_gpu.yaml` | Combined multi-modal |
| `toymix_bbbc047_gpu.yaml`, `toymix_bbbc047_filtered_gpu.yaml`, `toymix_rxrx3_gpu.yaml`, `toymix_rxrx3_dti_gpu.yaml`, `toymix_dti_*_gpu.yaml` | ToyMix multi-modal variants |
| `bbbc047_gpu.yaml` | BBBC047 standalone |

## For AI Agents

### Working In This Directory
- Keep files **thin** — only keys that are genuinely device-dependent (precision, num_workers, persistent_workers). Epochs, LR, scheduler belong in `../` one level up.
- Batch sizes defined here can still be overridden by `training/model/<dataset>_<model>.yaml` (later in composition) — check both places when batch size looks wrong.
- A new dataset/device combination must have a matching file here for the Hydra composition to work; missing files fall through to defaults and usually cause an OOM on GPU or a silently slow dataloader on CPU.

## Dependencies

### Internal
- Composes with `../<dataset>.yaml`, `../../accelerator/<device>.yaml`, and `../model/<dataset>_<model>.yaml`.

<!-- MANUAL: -->
