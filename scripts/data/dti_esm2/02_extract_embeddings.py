#!/usr/bin/env python
"""Stage 2: Extract ESM2 protein embeddings from amino acid sequences.

Takes the protein-drug CSV from Stage 1, extracts unique protein sequences,
and runs ESM2 (esm2_t36_3B_UR50D) to produce mean-pooled embeddings from
layer 36 (dimension 2560).

Each protein produces one .pt file in <output_dir>/embedding/ containing:
    {"mean_representations": {36: tensor([2560])}}

This stage uses the ``esm-extract`` CLI tool from the ``fair-esm`` package and
supports multi-GPU parallel processing via a thread-based GPU manager.
Intermediate FASTA files are written to <output_dir>/fasta/.

The process is idempotent: proteins that already have a .pt file are skipped.

Prerequisites:
    pip install fair-esm torch
    # Each GPU needs ~12 GB VRAM for the ESM2-3B model

Runtime: ~2-4 hours for 5,222 proteins on 2 GPUs.

Usage:
    python 02_extract_embeddings.py \\
        --input data/protein-drug.csv \\
        --output-dir data/protein-esm2 \\
        --gpus 0,1

    # Or use esm-extract directly for a single protein:
    # esm-extract esm2_t36_3B_UR50D protein.fasta output_dir/ \\
    #     --repr_layers 36 --include mean
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd


DEFAULT_ESM_MODEL = "esm2_t36_3B_UR50D"
DEFAULT_REPR_LAYER = 36
DEFAULT_EMBEDDING_DIM = 2560


class GPUManager:
    """Thread-safe round-robin GPU allocator."""

    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = gpu_ids
        self._idx = 0
        self._lock = Lock()

    def get_gpu(self) -> int:
        with self._lock:
            gpu = self.gpu_ids[self._idx % len(self.gpu_ids)]
            self._idx += 1
            return gpu


def validate_smiles(smiles_list: list[str]) -> list[str]:
    """Optionally validate SMILES with RDKit; returns only valid ones."""
    try:
        from rdkit import Chem

        valid = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
        print(f"  RDKit validation: {len(valid):,}/{len(smiles_list):,} valid SMILES")
        return valid
    except ImportError:
        print("  RDKit not available, skipping SMILES validation")
        return smiles_list


def write_fasta(protein_id: str, sequence: str, fasta_dir: str) -> str:
    """Write a single-sequence FASTA file and return its path."""
    fasta_path = os.path.join(fasta_dir, f"{protein_id}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">{protein_id}\n{sequence}\n")
    return fasta_path


def run_esm_extract(
    fasta_path: str, output_dir: str, gpu_id: int,
    esm_model: str, repr_layer: int,
) -> bool:
    """Run esm-extract for one FASTA file on a specific GPU.

    Returns True on success, False on failure.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "esm-extract",
        esm_model,
        fasta_path,
        output_dir,
        "--repr_layers",
        str(repr_layer),
        "--include",
        "mean",
    ]
    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR extracting {fasta_path}: {e.stderr[:200]}")
        return False
    except FileNotFoundError:
        print("  ERROR: esm-extract not found. Install with: pip install fair-esm")
        return False


def process_batch(
    proteins: list[tuple[str, str]],
    fasta_dir: str,
    embedding_dir: str,
    gpu_manager: GPUManager,
    esm_model: str,
    repr_layer: int,
):
    """Process a batch of proteins: write FASTA, run ESM, skip existing."""
    gpu_id = gpu_manager.get_gpu()
    for protein_id, sequence in proteins:
        pt_path = os.path.join(embedding_dir, f"{protein_id}.pt")
        if os.path.exists(pt_path):
            continue  # idempotent: skip already-extracted proteins

        fasta_path = write_fasta(protein_id, sequence, fasta_dir)
        run_esm_extract(fasta_path, embedding_dir, gpu_id, esm_model, repr_layer)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Extract ESM2 protein embeddings"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/protein-drug.csv",
        help="Input CSV from Stage 1 (default: data/protein-drug.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/protein-esm2",
        help="Output directory for FASTA and embedding files (default: data/protein-esm2)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs to use (default: '0')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of proteins per GPU batch (default: 32)",
    )
    parser.add_argument(
        "--esm-model",
        default=DEFAULT_ESM_MODEL,
        help=(
            "ESM-2 variant (default: %(default)s / 2560-d / layer 36). "
            "Use 'esm2_t33_650M_UR50D' with --repr-layer 33 for 1280-d "
            "(GRAM-DTI paper default)."
        ),
    )
    parser.add_argument(
        "--repr-layer",
        type=int,
        default=DEFAULT_REPR_LAYER,
        help="ESM-2 layer to extract (default: %(default)s for esm2_t36_3B_UR50D).",
    )
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    fasta_dir = os.path.join(args.output_dir, "fasta")
    embedding_dir = os.path.join(args.output_dir, "embedding")
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    print("Stage 2: Extracting ESM2 protein embeddings")
    print("=" * 60)
    print(f"  Model:      {args.esm_model}")
    print(f"  Repr layer: {args.repr_layer}")
    print(f"  GPUs:       {gpu_ids}")

    # Load DTI data and extract unique proteins
    print(f"\n  Loading {args.input} ...")
    df = pd.read_csv(args.input)
    proteins = (
        df[["Target_ID", "Target"]]
        .drop_duplicates(subset=["Target_ID"])
        .values.tolist()
    )
    print(f"  Unique proteins: {len(proteins):,}")

    # Check how many already have embeddings
    existing = sum(
        1
        for pid, _ in proteins
        if os.path.exists(os.path.join(embedding_dir, f"{pid}.pt"))
    )
    remaining = len(proteins) - existing
    print(f"  Already extracted: {existing:,}")
    print(f"  Remaining:         {remaining:,}")

    if remaining == 0:
        print("\n  All proteins already have embeddings. Nothing to do.")
        return

    # Optional SMILES sanity check — only runs when the CSV is a TDC-style
    # drug+target table. The GRAM-DTI protein collector emits protein-only CSVs.
    if "Drug" in df.columns:
        validate_smiles(df["Drug"].unique().tolist())

    # Process proteins in batches across GPUs
    gpu_manager = GPUManager(gpu_ids)
    batch_size = args.batch_size

    batches = [
        proteins[i : i + batch_size] for i in range(0, len(proteins), batch_size)
    ]
    print(f"\n  Processing {len(batches)} batches (batch_size={batch_size}) ...")

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = []
        for batch in batches:
            fut = executor.submit(
                process_batch, batch, fasta_dir, embedding_dir, gpu_manager,
                args.esm_model, args.repr_layer,
            )
            futures.append(fut)

        for i, fut in enumerate(futures):
            fut.result()
            if (i + 1) % 10 == 0 or (i + 1) == len(batches):
                print(f"    Completed batch {i + 1}/{len(batches)}")

    # Final count
    final_count = sum(
        1
        for pid, _ in proteins
        if os.path.exists(os.path.join(embedding_dir, f"{pid}.pt"))
    )
    print(f"\n  Embeddings extracted: {final_count:,}/{len(proteins):,}")
    print("Done.")


if __name__ == "__main__":
    main()
