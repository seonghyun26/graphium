from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import datamol as dm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphium.features.featurizer import mol_to_pyggraph
from graphium.trainer.predictor import PredictorModule


ADMET_TASKS = [
    "caco2_wang",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "bbb_martins",
    "ppbr_az",
    "vdss_lombardo",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "ld50_zhu",
    "herg",
    "ames",
    "dili",
]

PAIRMIXER_FEATURIZATION = {
    "atom_property_list_onehot": [
        "atomic-number",
        "group",
        "period",
        "total-valence",
        "hybridization",
        "chirality",
    ],
    "atom_property_list_float": [
        "degree",
        "formal-charge",
        "radical-electron",
        "aromatic",
        "in-ring",
        "mass",
        "electronegativity",
        "vdw-radius",
        "num-ring",
    ],
    "edge_property_list": ["bond-type-onehot", "stereo", "in-ring", "conjugated"],
    "add_self_loop": False,
    "explicit_H": False,
    "use_bonds_weights": False,
    "pos_encoding_as_features": {
        "pos_types": {
            "lap_eigvec": {
                "pos_level": "node",
                "pos_type": "laplacian_eigvec",
                "num_pos": 8,
                "normalization": "none",
                "disconnected_comp": True,
            },
            "lap_eigval": {
                "pos_level": "node",
                "pos_type": "laplacian_eigval",
                "num_pos": 8,
                "normalization": "none",
                "disconnected_comp": True,
            },
            "rw_pos": {
                "pos_level": "node",
                "pos_type": "rw_return_probs",
                "ksteps": 16,
            },
        }
    },
}


def load_admet_smiles(tasks: list[str] | None = None) -> list[str]:
    from tdc.benchmark_group import admet_group

    tasks = tasks or ADMET_TASKS
    group = admet_group(path=str(ROOT / "expts" / "data" / "admet"))
    all_smiles: set[str] = set()
    for task_name in tasks:
        benchmark = group.get(task_name)
        df = pd.concat([benchmark["train_val"], benchmark["test"]], ignore_index=True)
        all_smiles.update(df["Drug"].dropna().astype(str).tolist())
    return sorted(all_smiles)


def _pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    sq_norm = np.sum(X * X, axis=1, keepdims=True)
    dists = sq_norm + sq_norm.T - 2.0 * (X @ X.T)
    np.maximum(dists, 0.0, out=dists)
    return dists


def _knn_mask(X: np.ndarray, k: int) -> np.ndarray:
    n = X.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=bool)
    k_eff = max(1, min(k, n - 1))
    dists = _pairwise_squared_distances(X)
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, nn_idx] = True
    return mask


def _centered_linear_gram(X: np.ndarray) -> np.ndarray:
    gram = X @ X.T
    row_mean = gram.mean(axis=1, keepdims=True)
    return gram - row_mean


def _align_with_mask(X: np.ndarray, Y: np.ndarray, mask_xy: np.ndarray) -> float:
    n = X.shape[0]
    if n <= 1 or not mask_xy.any():
        return 0.0
    gx = _centered_linear_gram(X)
    gy = _centered_linear_gram(Y)
    return float((gx[mask_xy] * gy[mask_xy]).sum() / ((n - 1) ** 2))


def cknna(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of rows, got {X.shape} and {Y.shape}.")

    mask_x = _knn_mask(X, k)
    mask_y = _knn_mask(Y, k)
    align_xy = _align_with_mask(X, Y, mask_x & mask_y)
    align_xx = _align_with_mask(X, X, mask_x)
    align_yy = _align_with_mask(Y, Y, mask_y)
    denom = float(np.sqrt(max(align_xx, 0.0) * max(align_yy, 0.0)))
    if denom <= 1e-12:
        return 0.0
    return align_xy / denom


def pairwise_cknna(embeddings: dict[str, np.ndarray], k: int = 5) -> pd.DataFrame:
    keys = list(embeddings)
    matrix = np.zeros((len(keys), len(keys)), dtype=np.float32)
    for i, left in enumerate(keys):
        for j, right in enumerate(keys):
            matrix[i, j] = cknna(embeddings[left], embeddings[right], k=k)
    return pd.DataFrame(matrix, index=keys, columns=keys)


def _featurize_pairmixer_smiles(smiles: list[str]) -> tuple[list[Data], list[str]]:
    graphs: list[Data] = []
    valid_smiles: list[str] = []
    for smi in tqdm(smiles, desc="pairmixer featurize"):
        mol = dm.to_mol(smi)
        if mol is None or mol.GetNumAtoms() > 200:
            continue
        graph = mol_to_pyggraph(mol, **PAIRMIXER_FEATURIZATION)
        if isinstance(graph, Data):
            graphs.append(graph)
            valid_smiles.append(smi)
    return graphs, valid_smiles


@torch.no_grad()
def _extract_pairmixer_graph_embeddings(
    model: torch.nn.Module,
    graphs: list[Data],
    *,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    all_graph = []
    for start in tqdm(range(0, len(graphs), batch_size), desc="pairmixer forward"):
        batch_graphs = graphs[start : start + batch_size]
        batch = Batch.from_data_list(batch_graphs)
        for key in batch.keys():
            value = batch[key]
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.float16:
                    value = value.float()
                elif value.dtype == torch.int32:
                    value = value.long()
                batch[key] = value.to(device)
        batch = model.encoder_manager(batch)
        if model.pre_nn is not None:
            batch["feat"] = model.pre_nn(batch["feat"])
        if model.pre_nn_edges is not None:
            edge_feat = batch["edge_feat"]
            if torch.prod(torch.as_tensor(edge_feat.shape[:-1])) == 0:
                edge_feat = torch.zeros(
                    list(edge_feat.shape[:-1]) + [model.pre_nn_edges.out_dim],
                    device=edge_feat.device,
                    dtype=edge_feat.dtype,
                )
            else:
                edge_feat = model.pre_nn_edges(edge_feat)
            batch["edge_feat"] = edge_feat
        batch = model.gnn(batch)
        graph_output_nn = model.task_heads.graph_output_nn["graph"]
        graph_feat = graph_output_nn(batch).float()
        all_graph.append(graph_feat.cpu())
    if not all_graph:
        return np.zeros((0, 0), dtype=np.float32)
    return torch.cat(all_graph, dim=0).numpy().astype(np.float32)


def embed_pairmixer(
    smiles: list[str],
    *,
    checkpoint: str,
    cache_path: str | Path | None = None,
    batch_size: int = 32,
    device: str = "auto",
) -> tuple[list[str], np.ndarray]:
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["smiles"], np.asarray(payload["embeddings"], dtype=np.float32)

    device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    ckpt_path = ROOT / checkpoint
    predictor = PredictorModule.load_pretrained_model(str(ckpt_path), device="cpu")
    model = predictor.model.float().to(device)
    graphs, valid_smiles = _featurize_pairmixer_smiles(smiles)
    embeddings = _extract_pairmixer_graph_embeddings(model, graphs, batch_size=batch_size, device=device)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"smiles": valid_smiles, "embeddings": embeddings}, cache_path)
    return valid_smiles, embeddings


def embed_minimol(
    smiles: list[str],
    *,
    cache_path: str | Path,
    batch_size: int = 100,
) -> tuple[list[str], np.ndarray]:
    from scripts.minimol.minimol_eval import embed_smiles

    cache_path = Path(cache_path)
    embeddings = embed_smiles(smiles, cache_path=cache_path).astype(np.float32)
    return list(smiles), embeddings


def embed_mole(
    smiles: list[str],
    *,
    cache_path: str | Path,
    device: str = "cuda:0",
    batch_size: int = 256,
) -> tuple[list[str], np.ndarray]:
    from scripts.mole.mole_eval import embed_smiles

    cache_path = Path(cache_path)
    device = "cpu" if device == "auto" and not torch.cuda.is_available() else device
    embeddings = embed_smiles(smiles, cache_path=cache_path, device=device, batch_size=batch_size).astype(np.float32)
    return list(smiles), embeddings


def _resolve_kpgt_python() -> list[str]:
    env_python = Path("/home/shpark/miniforge3/envs/kpgt/bin/python")
    if env_python.exists():
        return [str(env_python)]
    mamba = shutil.which("mamba")
    if mamba is not None:
        return [mamba, "run", "-n", "kpgt", "python"]
    raise FileNotFoundError("KPGT env not found. Expected /home/shpark/miniforge3/envs/kpgt/bin/python or mamba.")


def embed_kpgt(
    smiles: list[str],
    *,
    cache_path: str | Path,
    device: str = "cuda:0",
    batch_size: int = 32,
    kpgt_repo: str = "downloads/kpgt/upstream/KPGT",
    checkpoint: str = "downloads/kpgt/base.pth",
) -> tuple[list[str], np.ndarray]:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".smi", delete=False) as handle:
        handle.write("\n".join(smiles))
        smiles_file = Path(handle.name)
    try:
        cmd = _resolve_kpgt_python() + [
            str(ROOT / "scripts" / "kpgt" / "kpgt_extract_embeddings.py"),
            "--kpgt-repo",
            str(ROOT / kpgt_repo),
            "--ckpt",
            str(ROOT / checkpoint),
            "--smiles-file",
            str(smiles_file),
            "--cache-path",
            str(cache_path),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
    finally:
        smiles_file.unlink(missing_ok=True)

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    survivor_smiles = [s for s in smiles if cache.get(s) is not None]
    embeddings = np.stack([np.asarray(cache[s], dtype=np.float32) for s in survivor_smiles], axis=0)
    return survivor_smiles, embeddings


def embed_qip(
    smiles: list[str],
    *,
    embedding_path: str | Path,
) -> tuple[list[str], np.ndarray]:
    path = Path(embedding_path)
    if not path.exists():
        raise FileNotFoundError(f"QIP embedding file not found: {path}")

    if path.suffix == ".npy":
        payload = np.load(path, allow_pickle=True).item()
    elif path.suffix == ".json":
        payload = json.loads(path.read_text())
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise ValueError("QIP embedding payload must be a dict mapping SMILES to vectors.")

    survivor_smiles = [s for s in smiles if s in payload]
    embeddings = np.stack([np.asarray(payload[s], dtype=np.float32) for s in survivor_smiles], axis=0)
    return survivor_smiles, embeddings


def _intersect_smiles(smiles_a: list[str], emb_a: np.ndarray, smiles_b: list[str], emb_b: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    idx_a = {s: i for i, s in enumerate(smiles_a)}
    idx_b = {s: i for i, s in enumerate(smiles_b)}
    shared = [s for s in smiles_a if s in idx_b]
    emb_a_shared = emb_a[[idx_a[s] for s in shared]]
    emb_b_shared = emb_b[[idx_b[s] for s in shared]]
    return shared, emb_a_shared, emb_b_shared


def align_embeddings_on_shared_smiles(model_embeddings: dict[str, tuple[list[str], np.ndarray]]) -> dict[str, np.ndarray]:
    shared_smiles: set[str] | None = None
    for smiles, _ in model_embeddings.values():
        smile_set = set(smiles)
        shared_smiles = smile_set if shared_smiles is None else shared_smiles & smile_set
    shared = sorted(shared_smiles or [])
    aligned: dict[str, np.ndarray] = {}
    for name, (smiles, emb) in model_embeddings.items():
        index = {s: i for i, s in enumerate(smiles)}
        aligned[name] = emb[[index[s] for s in shared]]
    return aligned


def load_model_embeddings(
    smiles: list[str],
    registry: dict[str, dict[str, Any]],
    *,
    cache_dir: str | Path = "results/cknna_model_alignment/cache",
) -> tuple[dict[str, tuple[list[str], np.ndarray]], pd.DataFrame]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, tuple[list[str], np.ndarray]] = {}
    rows: list[dict[str, Any]] = []

    for model_name, spec in registry.items():
        kind = spec["kind"]
        try:
            if kind == "minimol":
                out = embed_minimol(
                    smiles,
                    cache_path=cache_dir / f"{model_name}.pt",
                    batch_size=spec.get("batch_size", 100),
                )
            elif kind == "mole":
                out = embed_mole(
                    smiles,
                    cache_path=cache_dir / f"{model_name}.pt",
                    device=spec.get("device", "cuda:0"),
                    batch_size=spec.get("batch_size", 256),
                )
            elif kind == "kpgt":
                out = embed_kpgt(
                    smiles,
                    cache_path=cache_dir / f"{model_name}.pt",
                    device=spec.get("device", "cuda:0"),
                    batch_size=spec.get("batch_size", 32),
                    kpgt_repo=spec.get("kpgt_repo", "downloads/kpgt/upstream/KPGT"),
                    checkpoint=spec.get("checkpoint", "downloads/kpgt/base.pth"),
                )
            elif kind == "pairmixer":
                out = embed_pairmixer(
                    smiles,
                    checkpoint=spec["checkpoint"],
                    cache_path=cache_dir / f"{model_name}.pt",
                    batch_size=spec.get("batch_size", 32),
                    device=spec.get("device", "auto"),
                )
            elif kind == "qip":
                out = embed_qip(smiles, embedding_path=spec["embedding_path"])
            else:
                raise ValueError(f"Unsupported model kind: {kind}")

            model_smiles, emb = out
            outputs[model_name] = (model_smiles, emb)
            rows.append(
                {
                    "model": model_name,
                    "kind": kind,
                    "status": "ok",
                    "n_smiles": len(model_smiles),
                    "dim": emb.shape[1] if emb.ndim == 2 and emb.size else 0,
                    "note": "",
                }
            )
        except Exception as exc:  # notebook-facing surface; preserve partial progress
            rows.append(
                {
                    "model": model_name,
                    "kind": kind,
                    "status": "missing",
                    "n_smiles": 0,
                    "dim": 0,
                    "note": str(exc),
                }
            )
    return outputs, pd.DataFrame(rows)
