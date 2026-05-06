#!/usr/bin/env python
"""Extract per-epoch embeddings for the PairMixer 12M `toymix_all` run.

Companion to ``scripts/cka_analysis.py`` — that script only saves the
``last.ckpt`` embedding per pretrain case. This one loops over the saved
epoch checkpoints of the multi-modal-all run and writes one
``embeddings_toymix_all_ep{NNN}.pt`` per epoch, reusing the SMILES list
already cached in ``embeddings_toymix_all.pt`` so we don't re-hit TDC.

Run from the graphium/ dir:
    python scripts/cka_analysis_epochs.py
"""
import sys
import time
from pathlib import Path

import datamol as dm
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphium.features.featurizer import mol_to_pyggraph
from graphium.trainer.predictor import PredictorModule
from cka_analysis import FEATURIZATION, extract_embeddings  # noqa: E402

CKPT_DIR = Path(
    "models_checkpoints/toymix_dti_esmc_v2_lpm24_litopenai_bbbc047/"
    "pairmixer_12M/2026-05-01_01-59-58_20260501_015958"
)
# Epoch checkpoints saved during pretraining; full sweep gives a representational
# trajectory we can RSA against the foundation-model baselines.
EPOCH_CKPTS = {
    19:  "toymix_dti_esmc_v2_lpm24_litopenai_bbbc047_pairmixer_12M_epochepoch=019_20260501_015958.ckpt",
    39:  "toymix_dti_esmc_v2_lpm24_litopenai_bbbc047_pairmixer_12M_epochepoch=039_20260501_015958.ckpt",
    59:  "toymix_dti_esmc_v2_lpm24_litopenai_bbbc047_pairmixer_12M_epochepoch=059_20260501_015958.ckpt",
    79:  "toymix_dti_esmc_v2_lpm24_litopenai_bbbc047_pairmixer_12M_epochepoch=079_20260501_015958.ckpt",
    99:  "toymix_dti_esmc_v2_lpm24_litopenai_bbbc047_pairmixer_12M_epochepoch=099_20260501_015958.ckpt",
}

OUT_DIR = Path("results/cka_analysis")
SMILES_SOURCE = OUT_DIR / "embeddings_toymix_all.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not SMILES_SOURCE.exists():
        raise SystemExit(
            f"Missing {SMILES_SOURCE}. Run scripts/cka_analysis.py first to "
            "produce the canonical SMILES list."
        )

    src = torch.load(SMILES_SOURCE, map_location="cpu", weights_only=False)
    smiles = src["smiles"]
    print(f"Loaded {len(smiles)} SMILES from {SMILES_SOURCE.name}")

    print("Featurizing SMILES (one-shot, reused across all epoch ckpts)...")
    t0 = time.time()
    graphs = []
    for smi in tqdm(smiles, desc="  featurize"):
        mol = dm.to_mol(smi)
        if mol is None or mol.GetNumAtoms() > 200:
            graphs.append(None)
            continue
        g = mol_to_pyggraph(mol, **FEATURIZATION)
        graphs.append(g if isinstance(g, Data) else None)
    valid_idx = [i for i, g in enumerate(graphs) if g is not None]
    valid_graphs = [graphs[i] for i in valid_idx]
    valid_smiles = [smiles[i] for i in valid_idx]
    print(f"  {len(valid_graphs)}/{len(smiles)} graphs valid in {time.time() - t0:.1f}s")

    for epoch, fname in EPOCH_CKPTS.items():
        out_path = OUT_DIR / f"embeddings_toymix_all_ep{epoch:03d}.pt"
        if out_path.exists():
            print(f"\n[ep{epoch:03d}] cache exists, skipping ({out_path})")
            continue
        ckpt = CKPT_DIR / fname
        if not ckpt.exists():
            print(f"\n[ep{epoch:03d}] missing checkpoint: {ckpt}")
            continue
        print(f"\n[ep{epoch:03d}] {ckpt}")
        t0 = time.time()
        predictor = PredictorModule.load_pretrained_model(str(ckpt), device="cpu")
        model = predictor.model.float().to(device)
        model.eval()
        print(f"  model loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        gnn_emb, graph_emb = extract_embeddings(
            model, valid_graphs, batch_size=32, device=device,
        )
        print(f"  gnn={tuple(gnn_emb.shape)}  graph={tuple(graph_emb.shape)}  "
              f"({time.time() - t0:.1f}s)")
        torch.save(
            {"gnn": gnn_emb, "graph": graph_emb, "smiles": valid_smiles},
            out_path,
        )
        print(f"  -> {out_path}")

        del predictor, model, gnn_emb, graph_emb

    print("\nDone.")


if __name__ == "__main__":
    main()
