"""PairMixer (and any graphium-pretrained backbone) embedding adapter.

Loads a ``.ckpt`` via ``PredictorModule.load_pretrained_model``, then forwards
each SMILES through ``pre_nn -> gnn -> graph_output_nn['graph']`` to produce
the same graph-level embedding the in-training linear-probe callback uses.

Featurization goes through a Hydra-composed graphium datamodule so the atom /
bond / PE featurizer matches what the backbone was pretrained with.
"""
from __future__ import annotations

import contextlib
import hashlib
import sys
from functools import partial
from pathlib import Path
from typing import Any, List, Tuple

import joblib
import numpy as np
import torch
from tqdm import tqdm

from .base import MoleculeEncoder


ROOT = Path(__file__).resolve().parents[2]
HYDRA_CFG_DIR = ROOT / "expts" / "hydra-configs"


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    """Patch joblib so every completed delayed() ticks the tqdm bar."""
    class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()


_FEATURIZE_FIRST_ERROR_PRINTED = False


def _featurize_one(smiles: str, smiles_transformer) -> Any:
    """Run graphium's smiles_transformer. Reports the first failure's details
    to stderr — graphium can either raise OR return an error string, so both
    paths are surfaced."""
    global _FEATURIZE_FIRST_ERROR_PRINTED
    try:
        g = smiles_transformer(smiles, mask_nan=0.0)
    except Exception:
        if not _FEATURIZE_FIRST_ERROR_PRINTED:
            _FEATURIZE_FIRST_ERROR_PRINTED = True
            import sys, traceback
            print(
                f"\n  [pairmixer] first featurization EXCEPTION on SMILES={smiles!r}:",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
        return None
    if g is None or isinstance(g, str):
        if not _FEATURIZE_FIRST_ERROR_PRINTED:
            _FEATURIZE_FIRST_ERROR_PRINTED = True
            import sys
            print(
                f"\n  [pairmixer] first featurization SENTINEL on SMILES={smiles!r}:"
                f"\n      returned = {type(g).__name__}  value = {g!r}",
                file=sys.stderr,
            )
        return None
    return g


def _featurize_smiles(
    smiles: List[str], smiles_transformer, n_jobs: int,
) -> Tuple[List[Any], np.ndarray]:
    """Parallel graphium featurization. Returns (graphs_of_survivors, mask)."""
    from joblib import Parallel, delayed

    n_jobs = max(1, int(n_jobs))
    desc = f"featurizing (n_jobs={n_jobs})"
    if n_jobs == 1:
        graphs = [
            _featurize_one(s, smiles_transformer)
            for s in tqdm(smiles, desc=desc, unit="mol", smoothing=0.05)
        ]
    else:
        with _tqdm_joblib(tqdm(total=len(smiles), desc=desc, unit="mol", smoothing=0.05)):
            graphs = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_featurize_one)(s, smiles_transformer) for s in smiles
            )

    mask = np.array([g is not None for g in graphs], dtype=bool)
    kept = [g for g in graphs if g is not None]
    if (~mask).any():
        print(f"  [pairmixer] WARN: {int((~mask).sum()):,} SMILES failed featurization", flush=True)
    return kept, mask


def _build_datamodule(model_name: str) -> Any:
    """Hydra-compose dti_eval so we get the same featurizer the ckpt was trained with."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from graphium.config._loader import load_accelerator, load_datamodule

    with initialize_config_dir(version_base=None, config_dir=str(HYDRA_CFG_DIR)):
        cfg = compose(
            config_name="main",
            overrides=[
                f"model={model_name}",
                "accelerator=gpu",
                "tasks=dti_eval",
                "training=dti_eval",
                "++constants.dti_subset=DAVIS",
                "++constants.dti_split_method=random",
                "++constants.dti_split_seed=0",
                "++constants.seed=0",
            ],
        )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict, accelerator_type = load_accelerator(cfg_dict)
    return load_datamodule(cfg_dict, accelerator_type)


class PairMixerEncoder(MoleculeEncoder):
    """Embedding extractor for any graphium PredictorModule checkpoint.

    Despite the class name, this works for any graphium backbone whose
    ``task_heads.graph_output_nn['graph']`` produces a graph-level embedding —
    PairMixer, GPS++, GCN, etc. The name sticks for historical reasons.
    """

    def __init__(
        self, ckpt_path: str, *,
        model_name: str = "pairmixer_12M",
        device: str = "cuda:0",
        batch_size: int = 32,
        featurize_n_jobs: int = 8,
        ckpt_tag: str | None = None,
    ):
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.ckpt_path = str(ckpt_path)
        self.model_name = model_name
        self.device_str = device
        self.batch_size = batch_size
        self.featurize_n_jobs = featurize_n_jobs
        self.ckpt_tag = ckpt_tag or Path(ckpt_path).stem

        ckpt_hash = hashlib.sha256(self.ckpt_path.encode()).hexdigest()[:12]
        self.encoder_tag = f"{model_name}_{ckpt_hash}"
        self.out_dim = -1  # finalized on first forward pass (depends on head config)

        self._datamodule = None
        self._backbone = None
        self._torch_device = None

    def _ensure_backbone(self):
        if self._backbone is not None:
            return
        from graphium.trainer.predictor import PredictorModule

        self._datamodule = _build_datamodule(self.model_name)
        self._torch_device = torch.device(self.device_str)
        # PyTorch 2.6 flipped ``torch.load``'s default to ``weights_only=True``,
        # which rejects graphium's pickled-class ckpts. Our checkpoints are
        # trusted (we trained them), so force the legacy behavior just for this
        # load and restore torch.load immediately after.
        _orig_load = torch.load
        def _trusted_load(*args, **kwargs):
            # Force-override (not setdefault) because Lightning's cloud_io._load
            # passes ``weights_only=True`` explicitly, which setdefault wouldn't
            # replace. Safe within this narrowly-scoped monkey-patch.
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)
        torch.load = _trusted_load
        try:
            predictor = PredictorModule.load_pretrained_model(
                name_or_path=self.ckpt_path, device=str(self._torch_device),
            )
        finally:
            torch.load = _orig_load
        backbone = predictor.model
        backbone.to(self._torch_device)
        if "graph" not in backbone.task_heads.graph_output_nn:
            raise RuntimeError(
                "backbone.task_heads.graph_output_nn['graph'] missing — "
                f"checkpoint {self.ckpt_path} has no graph-level head."
            )
        self._backbone = backbone

    @property
    def metadata(self) -> dict:
        return {
            "model":     self.model_name,
            "ckpt_tag":  self.ckpt_tag,
            "ckpt_path": self.ckpt_path,
            "device":    self.device_str,
        }

    def _forward_graphs(self, graphs: List[Any]) -> torch.Tensor:
        """pre_nn -> (edges) -> gnn -> graph pooling -> graph_output_nn['graph']."""
        from graphium.data.collate import graphium_collate_fn
        from torch.utils.data import DataLoader

        backbone = self._backbone
        device = self._torch_device
        target_dtype = next(backbone.parameters()).dtype
        graph_output_nn = backbone.task_heads.graph_output_nn["graph"]

        loader = DataLoader(
            graphs, batch_size=self.batch_size, shuffle=False,
            collate_fn=partial(graphium_collate_fn, mask_nan=0), num_workers=0,
        )
        chunks: List[torch.Tensor] = []
        backbone.eval()
        n_batches = (len(graphs) + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for batch in tqdm(loader, total=n_batches, desc="GNN forward", unit="batch", smoothing=0.05):
                batch = batch.to(device)
                # Force float tensors to the backbone's dtype (some ckpts are fp16).
                keys = batch.keys() if callable(getattr(batch, "keys", None)) else list(batch.keys)
                for k in list(keys):
                    v = batch[k]
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        batch[k] = v.to(target_dtype)
                g = backbone.encoder_manager(batch)
                if backbone.pre_nn is not None:
                    g["feat"] = backbone.pre_nn.forward(g["feat"])
                if backbone.pre_nn_edges is not None:
                    e = g["edge_feat"]
                    if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
                        e = torch.zeros(
                            list(e.shape[:-1]) + [backbone.pre_nn_edges.out_dim],
                            device=e.device, dtype=e.dtype,
                        )
                    else:
                        e = backbone.pre_nn_edges.forward(e)
                    g["edge_feat"] = e
                g = backbone.gnn.forward(g)
                if backbone.gnn_layer_pooling is not None:
                    g["feat"] = backbone.gnn_layer_pooling(backbone.gnn._readout_cache)
                z = graph_output_nn(g)
                chunks.append(z.detach().float().cpu())
        return torch.cat(chunks, dim=0)

    def extract(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_backbone()
        graphs, mask = _featurize_smiles(
            smiles, self._datamodule.smiles_transformer, self.featurize_n_jobs,
        )
        if not graphs:
            if self.out_dim < 0:
                # Haven't inferred yet — can't produce a correct zero-shape array.
                raise RuntimeError(
                    "PairMixerEncoder.extract called with 0 survivors before "
                    "out_dim could be inferred. Pass at least one valid SMILES first."
                )
            return np.zeros((0, self.out_dim), dtype=np.float32), mask

        z = self._forward_graphs(graphs).numpy().astype(np.float32)
        if self.out_dim < 0:
            self.out_dim = int(z.shape[1])
        return z, mask
