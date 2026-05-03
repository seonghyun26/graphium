#!/usr/bin/env python
"""KPGT embedding extractor — runs INSIDE the dedicated `kpgt` conda env.

The graphium env can't host KPGT (DGL pinned vs torch 2.7 ABI mismatch), so the
adapter (``downstream/model/kpgt.py``) shells out to this script, which:

  1. Reads a newline-delimited SMILES file (``--smiles-file``).
  2. Featurizes each SMILES via ``smiles_to_graph_tune`` + ``RDKit2DNormalized``
     + ``Chem.RDKFingerprint(minPath=1, maxPath=7, fpSize=512)`` — the same
     pipeline as KPGT's official ``preprocess_downstream_dataset.py``.
  3. Forwards through ``LiGhTPredictor.generate_fps`` to produce a 2304-d
     embedding (3 × d_g_feats; d_g_feats=768 for the released ``base.pth``).
  4. Updates a per-SMILES torch cache at ``--cache-path`` so reruns hit cache.

Failures (invalid SMILES, RDKit errors) are stored as ``None`` so the adapter
can build a survivor mask the same way the MiniMol / MolE encoders do.

Usage:
    python kpgt_extract_embeddings.py \
        --kpgt-repo  /path/to/KPGT \
        --ckpt       /path/to/base.pth \
        --smiles-file smiles.txt \
        --cache-path datacache/kpgt_embeddings/admet_caco2_wang.pt \
        --device     cuda:0 \
        --batch-size 32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")  # silence RDKit's per-molecule warnings


def _add_kpgt_to_path(repo_dir: Path) -> None:
    """Make ``from src.X import Y`` resolve to the upstream KPGT checkout."""
    repo_dir = Path(repo_dir).resolve()
    if not (repo_dir / "src" / "model" / "light.py").exists():
        sys.exit(f"KPGT repo not found at {repo_dir} — pass --kpgt-repo.")
    sys.path.insert(0, str(repo_dir))


def _build_model(ckpt_path: Path, device: torch.device) -> "LiGhTPredictor":  # noqa: F821
    """Instantiate LiGhT with the ``base`` config and load the figshare ckpt."""
    from src.model.light import LiGhTPredictor as LiGhT
    from src.model_config import config_dict
    from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES

    cfg = config_dict["base"]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = LiGhT(
        d_node_feats=cfg["d_node_feats"],
        d_edge_feats=cfg["d_edge_feats"],
        d_g_feats=cfg["d_g_feats"],
        d_hpath_ratio=cfg["d_hpath_ratio"],
        n_mol_layers=cfg["n_mol_layers"],
        path_length=cfg["path_length"],
        n_heads=cfg["n_heads"],
        n_ffn_dense_layers=cfg["n_ffn_dense_layers"],
        input_drop=0.0, attn_drop=0.0, feat_drop=0.0,
        n_node_types=vocab.vocab_size,
    ).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    # Pretrained ckpt is a DDP-wrapped state_dict; strip ``module.`` prefix.
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()
    return model


def _featurize_one(
    smiles: str, *, max_length: int, n_virtual_nodes: int, descriptor_gen,
) -> Optional[Tuple["dgl.DGLGraph", torch.Tensor, torch.Tensor]]:  # noqa: F821
    """Returns (graph, fp, md) or ``None`` on RDKit failure.

    Mirrors KPGT's ``preprocess_downstream_dataset.py`` exactly so embeddings
    match the upstream pipeline byte-for-byte.
    """
    from src.data.featurizer import smiles_to_graph_tune

    g = smiles_to_graph_tune(smiles, max_length=max_length, n_virtual_nodes=n_virtual_nodes)
    if g is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp_bits = list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512))
    except Exception:
        return None
    fp = torch.tensor(fp_bits, dtype=torch.float32)

    # RDKit2DNormalized.process(smiles) returns [is_valid, *features]; drop col 0
    # (the validity flag) to match upstream's ``arr[:, 1:]`` slice. NaNs → 0,
    # matching ``MoleculeDataset._pre_process``.
    md_full = descriptor_gen.process(smiles)
    if md_full is None:
        return None
    md_arr = np.asarray(md_full[1:], dtype=np.float32)
    md_arr = np.where(np.isnan(md_arr), 0.0, md_arr)
    md = torch.from_numpy(md_arr)

    return g, fp, md


def _batched_forward(
    model: "LiGhTPredictor",  # noqa: F821
    feats: List[Tuple["dgl.DGLGraph", torch.Tensor, torch.Tensor]],  # noqa: F821
    *, device: torch.device, batch_size: int,
) -> np.ndarray:
    """Stack features into batches and run ``model.generate_fps`` on each."""
    import dgl
    from src.data.collator import preprocess_batch_light

    out_chunks: List[torch.Tensor] = []
    n = len(feats)
    desc = "kpgt forward"
    bar = tqdm(total=n, desc=desc, unit="mol", smoothing=0.05) if n else None
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = feats[i:i + batch_size]
            graphs = [g for (g, _, _) in chunk]
            batched = dgl.batch(graphs)
            # KPGT's Collator_tune does this remap so per-graph path indices
            # become global indices into the batched graph's node list.
            batched.edata["path"][:, :] = preprocess_batch_light(
                batched.batch_num_nodes(), batched.batch_num_edges(),
                batched.edata["path"][:, :],
            )
            batched = batched.to(device)
            fps = torch.stack([fp for (_, fp, _) in chunk], dim=0).to(device)
            mds = torch.stack([md for (_, _, md) in chunk], dim=0).to(device)
            out = model.generate_fps(batched, fps, mds)  # (B, 3*d_g_feats)
            out_chunks.append(out.detach().float().cpu())
            if bar is not None:
                bar.update(len(chunk))
    if bar is not None:
        bar.close()
    if not out_chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return torch.cat(out_chunks, dim=0).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kpgt-repo", required=True,
                        help="Path to the KPGT upstream checkout (contains src/).")
    parser.add_argument("--ckpt", required=True, help="Path to base.pth.")
    parser.add_argument("--smiles-file", required=True,
                        help="Newline-delimited SMILES (one per line, blank lines skipped).")
    parser.add_argument("--cache-path", required=True,
                        help="Per-SMILES torch.save cache (Dict[str, np.ndarray | None]).")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-virtual-nodes", type=int, default=2,
                        help="KPGT finetune default; matches MoleculeDataset.")
    parser.add_argument("--path-length", type=int, default=5,
                        help="KPGT base config; do not change without retraining.")
    args = parser.parse_args()

    repo_dir = Path(args.kpgt_repo).resolve()
    _add_kpgt_to_path(repo_dir)

    device = torch.device(args.device)
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, Optional[np.ndarray]] = (
        torch.load(str(cache_path), map_location="cpu", weights_only=False)
        if cache_path.exists() else {}
    )

    smiles_input = [
        s.strip() for s in Path(args.smiles_file).read_text().splitlines()
        if s.strip()
    ]
    # Dedup but keep first-seen order.
    smiles_unique = list(dict.fromkeys(smiles_input))
    missing = [s for s in smiles_unique if s not in cache]
    print(
        f"[kpgt] {len(smiles_unique):,} unique SMILES "
        f"({len(missing):,} missing, {len(smiles_unique) - len(missing):,} cached)",
        flush=True,
    )
    if not missing:
        print(f"[kpgt] cache fully covers input — nothing to embed.")
        return

    # KPGT vendors its own copy of rdNormalizedDescriptors which references
    # scipy.stats.gilbrat — renamed to ``gibrat`` in scipy 1.11+. Use the pip
    # ``descriptastorus`` package instead; it produces the same 200-d vector.
    from descriptastorus.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
    descriptor_gen = RDKit2DNormalized()

    print(f"[kpgt] loading checkpoint from {args.ckpt}", flush=True)
    model = _build_model(Path(args.ckpt), device)

    print(f"[kpgt] featurizing {len(missing):,} SMILES…", flush=True)
    t0 = time.time()
    feats: List[Tuple] = []
    survivor_smiles: List[str] = []
    failed: List[str] = []
    for s in tqdm(missing, desc="featurize", unit="mol", smoothing=0.05):
        try:
            r = _featurize_one(
                s, max_length=args.path_length,
                n_virtual_nodes=args.n_virtual_nodes,
                descriptor_gen=descriptor_gen,
            )
        except Exception:
            r = None
        if r is None:
            failed.append(s)
        else:
            survivor_smiles.append(s)
            feats.append(r)
    print(
        f"[kpgt] featurization: {len(survivor_smiles):,} ok, {len(failed):,} failed "
        f"({time.time() - t0:.1f}s)", flush=True,
    )

    if feats:
        embeddings = _batched_forward(
            model, feats, device=device, batch_size=args.batch_size,
        )
    else:
        embeddings = np.zeros((0, 2304), dtype=np.float32)

    # Persist: failures get None sentinels; survivors get their float32 vectors.
    for s in failed:
        cache[s] = None
    for s, vec in zip(survivor_smiles, embeddings):
        cache[s] = np.asarray(vec, dtype=np.float32)

    torch.save(cache, str(cache_path))
    dim = embeddings.shape[1] if embeddings.size else "?"
    print(
        f"[kpgt] wrote cache → {cache_path} "
        f"(survivors={len(survivor_smiles):,} dim={dim} failed={len(failed):,})",
        flush=True,
    )


if __name__ == "__main__":
    main()
