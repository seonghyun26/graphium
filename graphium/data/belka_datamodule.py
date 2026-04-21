"""
SKELETON — BELKA downstream data module for graphium.

This is a draft that covers the plumbing (HF download, subsample, splits, per-target
task_specific_args) but **deliberately leaves two things under-tested**:

  1. The Kaggle "novel building blocks" OOD split. The HF mirrors used here carry only
     the training parquets. The official OOD test split comes from the Kaggle test set
     (`test.csv`, ~1.67M rows with building-block overlap flags). If you want the OOD
     benchmark, drop the raw Kaggle CSV under `belka_cache_dir/kaggle_test.csv` and set
     `split_kind="kaggle_ood"` — see `_load_kaggle_ood_split`.
  2. Class balancing. Positives are ~0.5–1% per target. The skeleton passes raw labels
     through; for a real run you likely want `class_weight` in the loss or a weighted
     sampler.

Source options per target:
  - HuggingFace `phanvancongthanh/belka-protein-{brd4,hsa,seh}` (21 parquet shards each,
    SMILES + binary label in `y`).
  - Kaggle `leash-bio/leash-BELKA` raw CSV (all 3 targets in one long table, needs
    pivot). Not wired up here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import omegaconf
import pandas as pd
from loguru import logger

from graphium.utils import fs
from graphium.data.datamodule import (
    DatasetProcessingParams,
    MultitaskFromSmilesDataModule,
)


class BelkaBenchmarkDataModule(MultitaskFromSmilesDataModule):
    """
    Downstream benchmark on a subsample of BELKA (Leash Bio, 2024 Kaggle competition).

    Three binary binding tasks share one compound set:
      belka_brd4, belka_hsa, belka_seh.

    Parameters:
        belka_targets: subset of ['brd4','hsa','seh']; default = all three.
        subsample_size: number of rows per target to draw (default 200_000). Set None
            to keep everything (~98M rows per target — only for pretraining scale).
        split_kind: 'random' | 'scaffold' | 'kaggle_ood'. See docstring.
        subsample_seed: RNG seed for subsampling.
        val_fraction / test_fraction: for 'random'/'scaffold' splits.
        belka_cache_dir: where raw parquet shards land.
        hf_repo_template: override if you're mirroring the files somewhere else.
    """

    _DEFAULT_TARGETS = ("brd4", "hsa", "seh")
    _DEFAULT_CACHE = "/home/shpark/prj-molrepr/datacache/belka"
    _HF_REPO = "phanvancongthanh/belka-protein-{target}"

    def __init__(
        self,
        belka_targets: Optional[Union[str, List[str]]] = None,
        subsample_size: Optional[int] = 200_000,
        split_kind: str = "random",
        subsample_seed: int = 0,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        belka_cache_dir: Optional[Union[str, Path]] = None,
        hf_repo_template: Optional[str] = None,
        # --- MultitaskFromSmilesDataModule passthrough ---
        processed_graph_data_path: Optional[Union[str, Path]] = None,
        dataloading_from: str = "ram",
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        batch_size_training: int = 64,
        batch_size_inference: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        multiprocessing_context: Optional[str] = None,
        featurization_n_jobs: int = -1,
        featurization_progress: bool = False,
        featurization_backend: str = "loky",
        collate_fn: Optional[Callable] = None,
        prepare_dict_or_graph: str = "pyg:graph",
        **kwargs,
    ):
        cache_dir = str(belka_cache_dir or self._DEFAULT_CACHE)
        fs.mkdir(cache_dir, exist_ok=True)
        repo_template = hf_repo_template or self._HF_REPO

        if belka_targets is None:
            belka_targets = list(self._DEFAULT_TARGETS)
        if isinstance(belka_targets, str):
            belka_targets = [belka_targets]
        unknown = [t for t in belka_targets if t not in self._DEFAULT_TARGETS]
        if unknown:
            raise ValueError(f"Unknown BELKA target(s): {unknown}")

        task_specific_args = {}
        for tgt in belka_targets:
            df = self._load_target_parquet(tgt, cache_dir, repo_template)
            if subsample_size is not None and len(df) > subsample_size:
                df = self._stratified_subsample(df, subsample_size, subsample_seed)
            split_path = self._build_split(
                tgt, df, split_kind, cache_dir, subsample_seed, val_fraction, test_fraction
            )
            task_specific_args[f"belka_{tgt}"] = DatasetProcessingParams(
                df=df.rename(columns={"smiles": "SMILES", "y": "Y"}),
                smiles_col="SMILES",
                label_cols=["Y"],
                splits_path=split_path,
                split_names=["train", "val", "test"],
                task_level="graph",
            )

        super().__init__(
            task_specific_args=task_specific_args,
            featurization=featurization,
            processed_graph_data_path=processed_graph_data_path,
            dataloading_from=dataloading_from,
            batch_size_training=batch_size_training,
            batch_size_inference=batch_size_inference,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            featurization_n_jobs=featurization_n_jobs,
            featurization_progress=featurization_progress,
            featurization_backend=featurization_backend,
            collate_fn=collate_fn,
            prepare_dict_or_graph=prepare_dict_or_graph,
            **kwargs,
        )

    @classmethod
    def _load_target_parquet(cls, target: str, cache_dir: str, repo_template: str) -> pd.DataFrame:
        """Download (once) and concatenate the 21 HF parquet shards for one target."""
        target_dir = fs.join(cache_dir, target)
        fs.mkdir(target_dir, exist_ok=True)
        combined_path = fs.join(target_dir, "combined.parquet")
        if fs.exists(combined_path):
            return pd.read_parquet(combined_path)

        # TODO: replace with `huggingface_hub.snapshot_download` for robustness.
        import httpx

        repo = repo_template.format(target=target)
        shards = []
        for i in range(21):
            shard_name = f"train-{i:05d}-of-00021.parquet"
            dest = fs.join(target_dir, shard_name)
            if not fs.exists(dest):
                url = f"https://huggingface.co/datasets/{repo}/resolve/main/data/{shard_name}"
                logger.info(f"Downloading {url} -> {dest}")
                with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_bytes():
                            f.write(chunk)
            shards.append(pd.read_parquet(dest))
        df = pd.concat(shards, ignore_index=True)
        df.to_parquet(combined_path)
        # Free per-shard files to save disk (optional; comment out if you prefer caching).
        # for i in range(21): fs.rm(fs.join(target_dir, f"train-{i:05d}-of-00021.parquet"))
        return df

    @staticmethod
    def _stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        """Keep positive/negative ratio stable after subsampling."""
        rng = np.random.default_rng(seed)
        pos = df[df["y"] == 1]
        neg = df[df["y"] == 0]
        pos_frac = len(pos) / len(df)
        n_pos = int(round(n * pos_frac))
        n_neg = n - n_pos
        pos_samp = pos.sample(n=min(n_pos, len(pos)), random_state=rng.integers(1 << 30))
        neg_samp = neg.sample(n=min(n_neg, len(neg)), random_state=rng.integers(1 << 30))
        return pd.concat([pos_samp, neg_samp]).sample(frac=1, random_state=seed).reset_index(drop=True)

    def _build_split(
        self,
        target: str,
        df: pd.DataFrame,
        split_kind: str,
        cache_dir: str,
        seed: int,
        val_fraction: float,
        test_fraction: float,
    ) -> str:
        if split_kind == "random":
            indices = self._random_split(len(df), seed, val_fraction, test_fraction)
        elif split_kind == "scaffold":
            # TODO: Murcko scaffold split via rdkit.Chem.Scaffolds.MurckoScaffold; omitted
            # in the skeleton because it's a few hundred lines of its own.
            raise NotImplementedError("scaffold split not implemented yet")
        elif split_kind == "kaggle_ood":
            # TODO: wire up once the Kaggle test CSV is staged under belka_cache_dir.
            raise NotImplementedError("kaggle_ood split not implemented yet")
        else:
            raise ValueError(f"Unknown split_kind={split_kind}")

        split_df = self._indices_to_split_csv(indices)
        split_path = fs.join(cache_dir, f"{target}_{split_kind}_seed{seed}_split.csv")
        split_df.to_csv(split_path, index=False)
        return split_path

    @staticmethod
    def _random_split(n: int, seed: int, val_f: float, test_f: float):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_test = int(round(n * test_f))
        n_val = int(round(n * val_f))
        return {
            "train": perm[: n - n_val - n_test].tolist(),
            "val": perm[n - n_val - n_test : n - n_test].tolist(),
            "test": perm[n - n_test :].tolist(),
        }

    @staticmethod
    def _indices_to_split_csv(indices: Dict[str, List[int]]) -> pd.DataFrame:
        max_len = max(len(v) for v in indices.values())
        out = {}
        for k, v in indices.items():
            pad = [float("nan")] * (max_len - len(v))
            out[k] = list(v) + pad
        return pd.DataFrame(out)
