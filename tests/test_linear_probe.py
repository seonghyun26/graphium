"""Unit tests for the ADMET linear-probe callback helpers.

These tests exercise the fit/score/spearman math without needing a real
backbone or TDC download. Full end-to-end verification is done via the
smoke pretrain run.
"""

import torch

from graphium.finetuning.linear_probe import (
    ADMETLinearProbeCallback,
    _fit_linear_head,
    _score_linear_head,
    _spearman,
)


def _make_callback(**overrides):
    cfg = {
        "enabled": True,
        "tasks": [
            {"name": "caco2_wang", "kind": "regression", "metric": "mae"},
            {"name": "herg", "kind": "classification", "metric": "auroc"},
        ],
        "max_epochs": 50,
        "batch_size": 32,
        "lr": 5e-2,
        "weight_decay": 0.0,
        "embedding_level": "graph",
        "every_n_val_epochs": 1,
        "tdc_cache_dir": None,
        "probe_seed": 0,
    }
    cfg.update(overrides)
    return ADMETLinearProbeCallback(cfg)


def test_callback_instantiation_validates_kinds_and_metrics():
    cb = _make_callback()
    assert cb.max_epochs == 50
    assert len(cb.tasks) == 2

    try:
        _make_callback(tasks=[{"name": "x", "kind": "bogus", "metric": "mae"}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad kind")

    try:
        _make_callback(tasks=[{"name": "x", "kind": "regression", "metric": "bogus"}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad metric")


def test_fit_and_score_regression_converges_on_linear_data():
    torch.manual_seed(0)
    d = 8
    n = 512
    true_w = torch.randn(d)
    z = torch.randn(n, d)
    y = z @ true_w + 0.01 * torch.randn(n)

    head = torch.nn.Linear(d, 1)
    _fit_linear_head(
        head=head,
        z=z,
        y=y,
        kind="regression",
        max_epochs=200,
        batch_size=64,
        lr=1e-1,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )
    mae = _score_linear_head(
        head=head,
        z=z,
        y=y,
        kind="regression",
        metric="mae",
        batch_size=64,
        device=torch.device("cpu"),
    )
    # A perfectly linear fit should reach low MAE.
    assert mae < 0.2, f"MAE too high: {mae}"


def test_fit_and_score_classification_converges_on_separable_data():
    torch.manual_seed(0)
    d = 4
    n = 512
    w = torch.tensor([1.0, -1.0, 0.5, 0.0])
    z = torch.randn(n, d)
    logits = z @ w
    y = (logits > 0).float()

    head = torch.nn.Linear(d, 1)
    _fit_linear_head(
        head=head,
        z=z,
        y=y,
        kind="classification",
        max_epochs=200,
        batch_size=64,
        lr=1e-1,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )
    auroc = _score_linear_head(
        head=head,
        z=z,
        y=y,
        kind="classification",
        metric="auroc",
        batch_size=64,
        device=torch.device("cpu"),
    )
    assert auroc > 0.9, f"AUROC too low: {auroc}"


def test_spearman_handles_monotonic_and_ties():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0])
    assert abs(_spearman(x, y) - 1.0) < 1e-6

    # Perfect anti-correlation
    assert abs(_spearman(x, -y) + 1.0) < 1e-6

    # Ties should not blow up
    x_ties = torch.tensor([1.0, 1.0, 2.0, 3.0])
    y_ties = torch.tensor([4.0, 5.0, 6.0, 7.0])
    assert -1.0 <= _spearman(x_ties, y_ties) <= 1.0
