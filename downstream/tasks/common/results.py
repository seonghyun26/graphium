"""Single-CSV-per-task results sink.

Open-schema append: unknown columns in a row are added to the header on the
fly, so encoder-specific metadata (ckpt_tag, feature_dim_mol, etc.) slots in
without a migration step. Matches ``graphium/cli/train_finetune_test.py``'s
behavior so the dashboard notebook ingests downstream rows the same way it
ingests ``experiment_results.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def append_result_row(csv_path: str | Path, row: Dict[str, Any]) -> None:
    """Append one row. Creates the file + expands columns as needed."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        merged = pd.concat([prev, new_df], ignore_index=True, sort=False)
        merged.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)
