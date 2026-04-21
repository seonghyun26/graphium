"""Shared utilities for downstream-task eval.

Exposes metrics, head trainers, and a tiny CSV-append helper. Everything here
is task-agnostic; task-specific logic lives in each ``downstream.tasks.<task>``
subpackage.
"""
from .heads import HEAD_CHOICES, train_head
from .metrics import classification_metrics, regression_metrics
from .results import append_result_row

__all__ = [
    "HEAD_CHOICES",
    "train_head",
    "classification_metrics",
    "regression_metrics",
    "append_result_row",
]
