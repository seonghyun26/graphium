<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# graphium/trainer

## Purpose
PyTorch Lightning wrapper and the training-side building blocks it needs: the `PredictorModule`, multi-task loss functions with NaN masking, metric implementations, predictor option dataclasses, and per-task summary trackers.

## Key Files

| File | Description |
|------|-------------|
| `predictor.py` | `PredictorModule` — the `LightningModule` that owns the architecture, optimizer, scheduler, loss, and metric logic |
| `losses.py` | Multi-task loss functions (MSE/MAE/BCE/CE) with NaN masking and per-task weighting |
| `metrics.py` | Torchmetrics wrappers plus custom metrics (MAE, MAPE, AUROC, AP, Pearson/Spearman, etc.) |
| `predictor_options.py` | Dataclasses for predictor configuration (eval intervals, min/max tracking, EMA) |
| `predictor_summaries.py` | Running summaries used for W&B logging and stdout progress |
| `__init__.py` | Re-exports `PredictorModule` |
| `README.md` | Upstream folder guide |

## For AI Agents

### Working In This Directory
- **NaN masking is non-negotiable**: missing targets in multi-task batches must not contribute to loss or metrics. All new losses/metrics must handle NaN inputs explicitly — mirror the pattern in `losses.py::loss_fn_with_nan`.
- `PredictorModule.training_step`/`validation_step`/`test_step` dispatch per task and aggregate; extending to a new graph level means updating the multi-level loop here, not in the architecture.
- Scheduler selection is resolved by name via `graphium/utils/spaces.py::SCHEDULER_DICT` — register new schedulers there.
- Running stats for `on_train_epoch_end` are accumulated in `predictor_summaries.py`. Keep this lightweight — heavy logging should go through W&B callbacks instead.

### Testing Requirements
- `tests/test_predictor.py`, `tests/test_losses.py`, `tests/test_metrics.py`, `tests/test_training.py`.
- IPU-equivalents live in `tests/test_ipu_losses.py` and `tests/test_ipu_metrics.py`.

### Common Patterns
- `PredictorModule.forward(batch)` must accept a `pyg.Batch` produced by `graphium_collate_fn`.
- Loss/metric classes accept a target mask and always return scalar tensors (never Python floats) so Lightning can aggregate across devices.

## Dependencies

### Internal
- `graphium/nn/architectures/global_architectures.py` — the wrapped network
- `graphium/utils/spaces.py` — loss/metric/scheduler registry
- `graphium/ipu/ipu_losses.py`, `graphium/ipu/ipu_metrics.py` — IPU mirrors

### External
- `torch`, `pytorch_lightning`, `torchmetrics`, `mup`

<!-- MANUAL: -->
