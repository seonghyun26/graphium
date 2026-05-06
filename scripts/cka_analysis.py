#!/usr/bin/env python
"""
CKA (Centered Kernel Alignment) analysis between pretrained model embeddings
and ADMET task labels.

Compares graph-level embeddings from GPS++ 800M models pretrained on different
datasets (toymix, dti, bbbc047) against all 22 ADMET benchmark task labels.

Extracts two embedding types:
  - gnn: GNN output (node-level, mean-pooled to graph-level)
  - graph: graph_output_nn output (after pooling MLP)

Computes both linear CKA and kernel CKA (RBF).

Usage:
    python scripts/cka_analysis.py [--output-dir results/cka_analysis] [--batch-size 32]
"""

import argparse
import sys
import time
from pathlib import Path

import datamol as dm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphium.features.featurizer import mol_to_pyggraph
from graphium.trainer.predictor import PredictorModule

# ── Configuration ─────────────────────────────────────────────────────────────

# Pretrained checkpoints — most recent run per pre-training dataset case
# (PairMixer 12M backbone for consistent comparison across modalities).
#
# Columns:
#   toymix                  — uni-modal molecular baseline (ToyMix only)
#   toymix_dti_esmcv2       — + DTI head w/ ESMCv2 protein embeddings
#   toymix_bbbc047          — + BBBC047 cell-morphology head
#   toymix_lpm24_litopenai  — + LPM-24 head w/ OpenAI text-embedding-3-small
#   toymix_all              — all of the above combined
CHECKPOINTS_UNIMODAL = {
    "toymix": "models_checkpoints/small-dataset/pairmixer_12M/"
              "2026-05-02_01-02-56_20260502_010256/"
              "toymix_pairmixer_12M_epochepoch=099_20260502_010256.ckpt",
}

CHECKPOINTS_MULTIMODAL = {
    "toymix_dti_esmcv2": "models_checkpoints/toymix-dti-esmc-v2/pairmixer_12M/"
                         "2026-04-28_01-54-53_20260428_015453/last.ckpt",
    "toymix_bbbc047": "models_checkpoints/toymix_bbbc047/pairmixer_12M/"
                      "2026-04-18_11-20-55_20260418_112055/last.ckpt",
    "toymix_lpm24_litopenai": "models_checkpoints/toymix-lpm24-litopenai/pairmixer_12M/"
                              "2026-04-24_22-23-52_20260424_222352/last.ckpt",
    # "toymix_all": "models_checkpoints/toymix_dti_esmc_v2_lpm24_litopenai_bbbc047/pairmixer_12M/"
    #               "2026-05-03_01-18-55_20260503_011855/last.ckpt",
    "toymix_all": "models_checkpoints/toymix_dti_esmc_v3_litmolformer_v2_bbbc047/pairmixer_12M_ema/2026-05-05_18-24-47_20260505_182447/toymix_dti_esmc_v3_litmolformer_v2_bbbc047_pairmixer_12M_ema_epochepoch=059_20260505_182447.ckpt"
}

CHECKPOINTS = {**CHECKPOINTS_UNIMODAL, **CHECKPOINTS_MULTIMODAL}

CHECKPOINT_MODALITY = {
    k: "uni-modal" for k in CHECKPOINTS_UNIMODAL
} | {
    k: "multi-modal" for k in CHECKPOINTS_MULTIMODAL
}

ADMET_TASKS = [
    # Absorption
    "caco2_wang", "hia_hou", "pgp_broccatelli", "bioavailability_ma",
    "lipophilicity_astrazeneca", "solubility_aqsoldb", "bbb_martins", "ppbr_az", "vdss_lombardo",
    # Metabolism
    "cyp2d6_veith", "cyp3a4_veith", "cyp2c9_veith",
    "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels", "cyp3a4_substrate_carbonmangels",
    # Excretion
    "half_life_obach", "clearance_hepatocyte_az", "clearance_microsome_az",
    # Toxicity
    "ld50_zhu", "herg", "ames", "dili",
]

ADMET_CATEGORIES = {
    "caco2_wang": "Absorption", "hia_hou": "Absorption",
    "pgp_broccatelli": "Absorption", "bioavailability_ma": "Absorption",
    "lipophilicity_astrazeneca": "Absorption", "solubility_aqsoldb": "Absorption",
    "bbb_martins": "Absorption", "ppbr_az": "Absorption", "vdss_lombardo": "Absorption",
    "cyp2d6_veith": "Metabolism", "cyp3a4_veith": "Metabolism", "cyp2c9_veith": "Metabolism",
    "cyp2c9_substrate_carbonmangels": "Metabolism",
    "cyp2d6_substrate_carbonmangels": "Metabolism",
    "cyp3a4_substrate_carbonmangels": "Metabolism",
    "half_life_obach": "Excretion",
    "clearance_hepatocyte_az": "Excretion", "clearance_microsome_az": "Excretion",
    "ld50_zhu": "Toxicity", "herg": "Toxicity", "ames": "Toxicity", "dili": "Toxicity",
}

TASK_TYPES = {
    "caco2_wang": "regression", "hia_hou": "classification",
    "pgp_broccatelli": "classification", "bioavailability_ma": "classification",
    "lipophilicity_astrazeneca": "regression", "solubility_aqsoldb": "regression",
    "bbb_martins": "classification", "ppbr_az": "regression", "vdss_lombardo": "regression",
    "cyp2d6_veith": "classification", "cyp3a4_veith": "classification",
    "cyp2c9_veith": "classification",
    "cyp2c9_substrate_carbonmangels": "classification",
    "cyp2d6_substrate_carbonmangels": "classification",
    "cyp3a4_substrate_carbonmangels": "classification",
    "half_life_obach": "regression",
    "clearance_hepatocyte_az": "regression", "clearance_microsome_az": "regression",
    "ld50_zhu": "regression", "herg": "classification",
    "ames": "classification", "dili": "classification",
}

# Featurization config — matches PairMixer 12M's expanded RDKit feature set
# (see expts/hydra-configs/model/pairmixer_12M.yaml). All 5 checkpoints in this
# analysis are PairMixer 12M, trained with this featurization.
FEATURIZATION = {
    "atom_property_list_onehot": [
        "atomic-number", "group", "period", "total-valence",
        "hybridization", "chirality",
    ],
    "atom_property_list_float": [
        "degree", "formal-charge", "radical-electron", "aromatic", "in-ring",
        "mass", "electronegativity", "vdw-radius", "num-ring",
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


# ── CKA functions ─────────────────────────────────────────────────────────────

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (n, d1) representation matrix
        Y: (n, d2) representation matrix (can be (n, 1) for labels)
    Returns:
        CKA similarity score (float)
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = (X.T @ Y).pow(2).sum()
    hsic_xx = (X.T @ X).pow(2).sum()
    hsic_yy = (Y.T @ Y).pow(2).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()


def _rbf_kernel(X: torch.Tensor) -> torch.Tensor:
    """Compute RBF kernel matrix with median heuristic for bandwidth."""
    sq_dists = torch.cdist(X, X, p=2).pow(2)
    mask = sq_dists > 0
    if mask.any():
        sigma = sq_dists[mask].median().sqrt()
    else:
        sigma = torch.tensor(1.0)
    return torch.exp(-sq_dists / (2 * sigma.pow(2) + 1e-10))


def kernel_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute kernel CKA with RBF kernels.

    Args:
        X: (n, d1) representation matrix
        Y: (n, d2) representation matrix
    Returns:
        CKA similarity score (float)
    """
    n = X.shape[0]
    K = _rbf_kernel(X)
    L = _rbf_kernel(Y)

    # Center kernels: H K H where H = I - 1/n
    H = torch.eye(n, device=X.device, dtype=X.dtype) - 1.0 / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    hsic_kl = (K_c * L_c).sum()
    hsic_kk = (K_c * K_c).sum()
    hsic_ll = (L_c * L_c).sum()

    denom = torch.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return (hsic_kl / denom).item()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_admet_data() -> dict:
    """Load all 22 ADMET benchmark datasets from TDC.

    Returns:
        dict mapping task_name -> {"smiles": list[str], "labels": np.ndarray}
    """
    from tdc.benchmark_group import admet_group

    group = admet_group(path="./data/tdc_admet_cache")

    admet_data = {}
    for task_name in ADMET_TASKS:
        benchmark = group.get(task_name)
        train_val = benchmark["train_val"]
        test = benchmark["test"]
        df = pd.concat([train_val, test], ignore_index=True)

        smiles = df["Drug"].tolist()
        labels = df["Y"].values.astype(np.float32)

        valid = ~np.isnan(labels)
        smiles = [s for s, v in zip(smiles, valid) if v]
        labels = labels[valid]

        admet_data[task_name] = {"smiles": smiles, "labels": labels}
        print(f"  {task_name}: {len(smiles)} molecules")

    return admet_data


def featurize_smiles(smiles_list: list) -> tuple:
    """Convert SMILES to PyG Data objects using graphium featurization.

    Returns:
        (graphs, valid_idx): list of Data objects + indices of successfully featurized SMILES
    """
    graphs = []
    valid_idx = []
    failed = 0
    skipped_large = 0
    for i, smi in enumerate(tqdm(smiles_list, desc="  Featurizing")):
        mol = dm.to_mol(smi)
        if mol is None:
            failed += 1
            continue
        # Skip very large molecules (Laplacian eigdecomp is O(n^3))
        if mol.GetNumAtoms() > 200:
            skipped_large += 1
            continue
        g = mol_to_pyggraph(mol, **FEATURIZATION)
        if isinstance(g, Data):
            graphs.append(g)
            valid_idx.append(i)
        else:
            failed += 1
    if failed:
        print(f"  Failed to featurize {failed}/{len(smiles_list)} molecules")
    if skipped_large:
        print(f"  Skipped {skipped_large} molecules with >200 atoms")
    return graphs, valid_idx


# ── Embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, graphs: list, batch_size: int = 32,
                       device: str = "cpu") -> tuple:
    """Extract GNN and graph-level embeddings from a pretrained model.

    Runs the model backbone (encoder_manager -> pre_nn -> gnn) and captures:
      - gnn: node features after GNN, mean-pooled to graph level
      - graph: output of graph_output_nn (pooling MLP)

    Returns:
        (gnn_embeddings, graph_embeddings) each of shape (n_graphs, dim)
    """
    model.eval()
    all_gnn = []
    all_graph = []

    for i in tqdm(range(0, len(graphs), batch_size), desc="  Extracting"):
        batch_graphs = graphs[i : i + batch_size]
        g = Batch.from_data_list(batch_graphs)

        # Cast dtypes: float16→float32, int32→int64; then move to device
        for key in g.keys():
            val = g[key]
            if isinstance(val, torch.Tensor):
                if val.dtype == torch.float16:
                    val = val.float()
                elif val.dtype == torch.int32:
                    val = val.long()
                g[key] = val.to(device)

        # Forward through backbone: encoder -> pre_nn -> gnn
        g = model.encoder_manager(g)

        if model.pre_nn is not None:
            g["feat"] = model.pre_nn(g["feat"])

        if model.pre_nn_edges is not None:
            e = g["edge_feat"]
            if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
                e = torch.zeros(
                    list(e.shape[:-1]) + [model.pre_nn_edges.out_dim],
                    device=e.device, dtype=e.dtype,
                )
            else:
                e = model.pre_nn_edges(e)
            g["edge_feat"] = e

        g = model.gnn(g)

        # GNN output: mean-pool node features to graph level
        gnn_feat = global_mean_pool(g["feat"].float(), g.batch)  # (n_graphs, gnn_dim)
        all_gnn.append(gnn_feat.cpu())

        # Graph-level embedding: through graph_output_nn (pooling + MLP)
        graph_output_nn = model.task_heads.graph_output_nn["graph"]
        graph_feat = graph_output_nn(g).float()  # (n_graphs, graph_dim)
        all_graph.append(graph_feat.cpu())

    return torch.cat(all_gnn, dim=0), torch.cat(all_graph, dim=0)


def extract_data_embeddings(graphs: list, batch_size: int = 256) -> torch.Tensor:
    """Extract raw data embeddings: atom features mean-pooled to graph level.

    No model is involved - this is the raw featurization baseline.

    Returns:
        data_embeddings of shape (n_graphs, feat_dim)
    """
    all_data = []
    for i in tqdm(range(0, len(graphs), batch_size), desc="  Data embeddings"):
        batch_graphs = graphs[i : i + batch_size]
        g = Batch.from_data_list(batch_graphs)
        feat = g["feat"]
        if feat.dtype == torch.float16:
            feat = feat.float()
        pooled = global_mean_pool(feat, g.batch)  # (n_graphs, feat_dim)
        all_data.append(pooled.cpu())
    return torch.cat(all_data, dim=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CKA analysis: pretrained embeddings vs ADMET labels"
    )
    parser.add_argument(
        "--output-dir", default="results/cka_analysis",
        help="Output directory for results (default: results/cka_analysis)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for forward pass (default: 32)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device for forward pass: 'cpu', 'cuda', 'cuda:N', or 'auto' (default: auto)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ADMET data
    print("=" * 60)
    print("Step 1: Loading ADMET data from TDC")
    print("=" * 60)
    admet_data = load_admet_data()

    # 2. Featurize all unique SMILES across tasks
    print("\n" + "=" * 60)
    print("Step 2: Featurizing SMILES")
    print("=" * 60)
    all_smiles = sorted({s for td in admet_data.values() for s in td["smiles"]})
    print(f"  Total unique SMILES across all tasks: {len(all_smiles)}")

    graphs, valid_idx = featurize_smiles(all_smiles)
    valid_smiles = [all_smiles[i] for i in valid_idx]
    smiles_to_idx = {s: i for i, s in enumerate(valid_smiles)}
    print(f"  Valid graphs: {len(graphs)}")

    # 3. Raw data embeddings baseline (no model)
    print("\n" + "=" * 60)
    print("Step 3: Raw data embeddings (baseline)")
    print("=" * 60)
    data_emb = extract_data_embeddings(graphs)
    print(f"  Data embeddings: {data_emb.shape}")
    torch.save(
        {"data": data_emb, "smiles": valid_smiles},
        output_dir / "embeddings_data.pt",
    )

    results = []
    print("  Computing CKA for data embeddings...")
    for task_name in tqdm(ADMET_TASKS, desc="  CKA (data)"):
        task_data = admet_data[task_name]
        indices, labels = [], []
        for smi, lbl in zip(task_data["smiles"], task_data["labels"]):
            if smi in smiles_to_idx:
                indices.append(smiles_to_idx[smi])
                labels.append(lbl)
        if len(indices) < 10:
            continue
        idx_t = torch.tensor(indices, dtype=torch.long)
        Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        X = data_emb[idx_t].float()
        lcka = linear_cka(X, Y)
        kcka = kernel_cka(X, Y)
        results.append({
            "checkpoint": "data",
            "modality": "none",
            "task": task_name,
            "category": ADMET_CATEGORIES[task_name],
            "task_type": TASK_TYPES[task_name],
            "embedding": "data",
            "linear_cka": lcka,
            "kernel_cka": kcka,
            "n_molecules": len(indices),
        })

    # 4. For each checkpoint: extract model embeddings & compute CKA
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'=' * 60}")
        print(f"Step 4: Checkpoint '{ckpt_name}'")
        print(f"  Path: {ckpt_path}")
        print("=" * 60)

        t0 = time.time()
        print("  Loading model...")
        predictor = PredictorModule.load_pretrained_model(ckpt_path, device="cpu")
        model = predictor.model.float().to(args.device)
        model.eval()
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        gnn_emb, graph_emb = extract_embeddings(
            model, graphs, batch_size=args.batch_size, device=args.device,
        )
        print(f"  GNN embeddings:   {gnn_emb.shape}")
        print(f"  Graph embeddings: {graph_emb.shape}")
        print(f"  Extraction took {time.time() - t0:.1f}s")

        # Save embeddings for later use
        torch.save(
            {"gnn": gnn_emb, "graph": graph_emb, "smiles": valid_smiles},
            output_dir / f"embeddings_{ckpt_name}.pt",
        )

        # Compute CKA for each ADMET task
        print("  Computing CKA...")
        for task_name in tqdm(ADMET_TASKS, desc=f"  CKA ({ckpt_name})"):
            task_data = admet_data[task_name]

            # Map task molecules to embedding indices
            indices, labels = [], []
            for smi, lbl in zip(task_data["smiles"], task_data["labels"]):
                if smi in smiles_to_idx:
                    indices.append(smiles_to_idx[smi])
                    labels.append(lbl)

            if len(indices) < 10:
                print(f"    Skipping {task_name}: only {len(indices)} valid molecules")
                continue

            idx_t = torch.tensor(indices, dtype=torch.long)
            Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (n, 1)

            for emb_name, all_emb in [("gnn", gnn_emb), ("graph", graph_emb)]:
                X = all_emb[idx_t].float()  # (n, dim)

                lcka = linear_cka(X, Y)
                kcka = kernel_cka(X, Y)

                results.append({
                    "checkpoint": ckpt_name,
                    "modality": CHECKPOINT_MODALITY[ckpt_name],
                    "task": task_name,
                    "category": ADMET_CATEGORIES[task_name],
                    "task_type": TASK_TYPES[task_name],
                    "embedding": emb_name,
                    "linear_cka": lcka,
                    "kernel_cka": kcka,
                    "n_molecules": len(indices),
                })

        # Free memory
        del predictor, model, gnn_emb, graph_emb

    # 4. Save results
    df = pd.DataFrame(results)
    csv_path = output_dir / "cka_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {csv_path}")
    print("=" * 60)

    # Print summary table
    for emb in ["gnn", "graph"]:
        print(f"\n--- {emb.upper()} embedding ---")
        sub = df[df["embedding"] == emb]
        pivot_lin = sub.pivot(index="task", columns="checkpoint", values="linear_cka")
        pivot_kern = sub.pivot(index="task", columns="checkpoint", values="kernel_cka")
        print("\nLinear CKA:")
        print(pivot_lin.to_string(float_format="%.4f"))
        print("\nKernel CKA:")
        print(pivot_kern.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
