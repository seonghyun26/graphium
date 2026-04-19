"""Lightning callback to upload checkpoints to Hugging Face Hub after training."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


# Map internal model config names to display names for HF repos.
_MODEL_DISPLAY = {
    "pairmixer_12M": "PairMixer12M",
    "pairmixer_10M": "PairMixer10M",
    "pairmixer_20M": "PairMixer20M",
    "pairmixer_40M": "PairMixer40M",
    "pairmixer_auto": "PairMixerAuto",
    "pairmixer_boltz": "PairMixerBoltz",
    "gpspp_800M": "GPSpp800M",
    "gpspp": "GPSpp",
    "pairformer_17M": "Pairformer17M",
    "pairformer_52M": "Pairformer52M",
    "pairformer_boltz": "PairformerBoltz",
}


def _make_model_display_name(constants_name: str) -> str:
    """Convert e.g. 'toymix_pairmixer_12M' -> 'PairMixer12M_toymix'.

    Format: {Architecture}_{pretrain_dataset}
    """
    # Try to extract model and dataset from constants.name
    # Pattern: {dataset}_{model} e.g. toymix_pairmixer_12M
    for key, display in sorted(_MODEL_DISPLAY.items(), key=lambda x: -len(x[0])):
        if key in constants_name:
            dataset = constants_name.replace(key, "").strip("_")
            if dataset:
                return f"{display}_{dataset}"
            return display
    return constants_name


class HuggingFaceUploadCallback(Callback):
    """Upload the final checkpoint and metadata to a Hugging Face Hub repo.

    Creates a model repo named ``{hf_user}/{Architecture}_{dataset}`` and
    uploads:
      - The last checkpoint (``last.ckpt`` or the newest ``.ckpt`` in dirpath)
      - ``config.yaml``  -- resolved Hydra config
      - ``README.md``    -- auto-generated model card with run metadata

    Parameters
    ----------
    hf_user : str
        HF username or organisation.
    repo_name : str
        Repository name override. If None, derived from ``constants.name``.
    cfg : dict
        Resolved Hydra config dict (for metadata upload).
    private : bool
        Whether the repo should be private. Default ``True``.
    """

    def __init__(
        self,
        hf_user: str,
        repo_name: str,
        cfg: dict,
        private: bool = True,
    ):
        super().__init__()
        self.hf_user = hf_user
        self.repo_name = repo_name
        self.cfg = cfg
        self.private = private
        self.repo_id = f"{hf_user}/{repo_name}"
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._upload(trainer)

    def _upload(self, trainer: pl.Trainer) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.warning("huggingface_hub not installed -- skipping HF upload")
            return

        dirpath = trainer.checkpoint_callback.dirpath
        if dirpath is None:
            logger.warning("No checkpoint dirpath configured -- skipping HF upload")
            return

        ckpt_path = self._find_checkpoint(dirpath)
        if ckpt_path is None:
            logger.warning(f"No .ckpt found in {dirpath} -- skipping HF upload")
            return

        api = HfApi()

        logger.info(f"Creating/accessing HF repo: {self.repo_id}")
        api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            private=self.private,
            exist_ok=True,
        )

        # All files go into a datetime subfolder: {run_timestamp}/
        folder = self.run_timestamp

        # 1. Upload checkpoint
        ckpt_name = Path(ckpt_path).name
        upload_name = f"{self.repo_name}.ckpt"
        size_mb = Path(ckpt_path).stat().st_size / 1e6
        logger.info(f"Uploading {ckpt_name} to {folder}/{upload_name} ({size_mb:.0f} MB)")
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=f"{folder}/{upload_name}",
            repo_id=self.repo_id,
        )

        # 2. Upload resolved config
        config_path = Path(dirpath) / "config.yaml"
        try:
            from omegaconf import OmegaConf
            if hasattr(self.cfg, '_metadata'):
                config_str = OmegaConf.to_yaml(self.cfg)
            else:
                config_str = yaml.dump(
                    dict(self.cfg) if not isinstance(self.cfg, dict) else self.cfg,
                    default_flow_style=False,
                )
        except Exception:
            config_str = yaml.dump(self.cfg, default_flow_style=False)
        config_path.write_text(config_str)
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo=f"{folder}/config.yaml",
            repo_id=self.repo_id,
        )

        # 3. Upload val/test results if available
        for results_file in ["val_results.yaml", "test_results.yaml"]:
            for search_dir in [Path(dirpath).parent, Path(os.getcwd())]:
                rp = search_dir / results_file
                if rp.exists():
                    api.upload_file(
                        path_or_fileobj=str(rp),
                        path_in_repo=f"{folder}/{results_file}",
                        repo_id=self.repo_id,
                    )
                    break

        # 4. Generate and upload model card (at repo root)
        card = self._build_model_card(f"{folder}/{upload_name}")
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=self.repo_id,
        )

        logger.info(f"Upload complete: https://huggingface.co/{self.repo_id}")

    def _find_checkpoint(self, dirpath: str) -> Optional[str]:
        """Find the best checkpoint to upload: prefer last.ckpt, else newest."""
        dirpath = Path(dirpath)
        if not dirpath.exists():
            return None
        last = dirpath / "last.ckpt"
        if last.exists():
            return str(last)
        ckpts = sorted(dirpath.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(ckpts[0]) if ckpts else None

    def _build_model_card(self, ckpt_name: str) -> str:
        """Generate a minimal model card."""
        cfg = self.cfg
        constants = cfg.get("constants", {}) if isinstance(cfg, dict) else {}
        model_name = self.repo_name
        seed = constants.get("seed", "?")
        max_epochs = constants.get("max_epochs", "?")

        arch = cfg.get("architecture", {}) if isinstance(cfg, dict) else {}
        gnn = arch.get("gnn", {})
        depth = gnn.get("depth", "?")
        hidden = gnn.get("hidden_dims", "?")
        pair_dim = gnn.get("layer_kwargs", {}).get("pair_dim", "?")

        dm = cfg.get("datamodule", {}).get("args", {}) if isinstance(cfg, dict) else {}
        bs = dm.get("batch_size_training", "?")

        return f"""---
tags:
  - graphium
  - molecular-property-prediction
  - gnn
  - pairmixer
license: apache-2.0
---

# {model_name}

Pre-trained PairMixer checkpoint from the Graphium framework.

## Model Details

| Parameter | Value |
|-----------|-------|
| Checkpoint | `{ckpt_name}` |
| GNN depth | {depth} |
| Hidden dim | {hidden} |
| Pair dim | {pair_dim} |
| Batch size | {bs} |
| Max epochs | {max_epochs} |
| Seed | {seed} |

## Usage

```python
from graphium.trainer.predictor import PredictorModule

predictor = PredictorModule.load_from_checkpoint("{ckpt_name}")
```

## Training Config

See `config.yaml` in this repo for the full resolved Hydra configuration.
"""


def upload_checkpoint(
    ckpt_path: str,
    hf_user: str,
    model_name: str,
    dataset: str,
    private: bool = True,
):
    """Standalone upload function for manually uploading existing checkpoints.

    Parameters
    ----------
    ckpt_path : str
        Path to the .ckpt file.
    hf_user : str
        HF username or org.
    model_name : str
        Display model name (e.g. "PairMixer12M").
    dataset : str
        Pre-training dataset name (e.g. "toymix").
    private : bool
        Whether repo is private.
    """
    from huggingface_hub import HfApi

    repo_name = f"{model_name}_{dataset}"
    repo_id = f"{hf_user}/{repo_name}"
    upload_name = f"{repo_name}.ckpt"
    folder = datetime.now().strftime("%Y%m%d_%H%M%S")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    size_mb = Path(ckpt_path).stat().st_size / 1e6
    logger.info(f"Uploading {ckpt_path} to {folder}/{upload_name} ({size_mb:.0f} MB) in {repo_id}")
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo=f"{folder}/{upload_name}",
        repo_id=repo_id,
    )

    # Simple model card at repo root
    card = f"""---
tags:
  - graphium
  - molecular-property-prediction
  - gnn
  - pairmixer
license: apache-2.0
---

# {repo_name}

Pre-trained {model_name} checkpoint on {dataset} from the Graphium framework.

## Usage

```python
from graphium.trainer.predictor import PredictorModule

predictor = PredictorModule.load_from_checkpoint("{folder}/{upload_name}")
```
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
    )

    logger.info(f"Upload complete: https://huggingface.co/{repo_id}")
