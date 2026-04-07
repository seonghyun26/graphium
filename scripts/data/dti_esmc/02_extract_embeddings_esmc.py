#!/usr/bin/env python
"""Stage 2 (ESM-C): Extract ESM-C 600M protein embeddings from amino acid sequences.

Takes a protein CSV (e.g. protein-esm2.csv from the existing ESM-2 pipeline)
with columns protein_id and sequence, and runs ESM-C 600M to produce
mean-pooled embeddings (dimension 1152).

Each protein produces one .pt file in <output_dir>/embedding/ containing:
    {"mean_embedding": tensor([1152])}

This stage uses the EvolutionaryScale ``esm`` package (not ``fair-esm``) and
supports multi-GPU parallel processing via a thread-based GPU manager.

The process is idempotent: proteins that already have a .pt file are skipped.

Prerequisites:
    # Requires Python 3.12 -- use the esmc conda env
    conda activate esmc
    pip install esm torch

Runtime: Estimated ~1-2 hours for 5,222 proteins on 2 GPUs (600M is faster
than ESM2-3B).

Usage:
    python 02_extract_embeddings_esmc.py \\
        --input data/dti-scratch/protein-esm2.csv \\
        --output-dir data/protein-esmc \\
        --gpus 0,1
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pandas as pd
import torch


ESM_MODEL = "esmc_600m"
EMBEDDING_DIM = 1152


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


def load_model(device: str):
    """Load ESM-C 600M model onto the specified device."""
    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained(ESM_MODEL).to(device)
    model.eval()
    return model


MAX_SEQ_LEN = 2048  # Truncate very long sequences to avoid GPU OOM


def extract_embedding(model, sequence: str) -> torch.Tensor:
    """Extract mean-pooled embedding for a single protein sequence.

    The model returns per-residue embeddings of shape (1, L+2, 1152) where
    +2 accounts for BOS/EOS tokens. We mean-pool over residue positions
    (excluding BOS at index 0 and EOS at the last index) to get a single
    1152-dim vector.

    Returns:
        1-D tensor of shape [1152].
    """
    from esm.sdk.api import ESMProtein, LogitsConfig

    # Truncate very long sequences to avoid OOM
    if len(sequence) > MAX_SEQ_LEN:
        sequence = sequence[:MAX_SEQ_LEN]

    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(
        protein_tensor,
        LogitsConfig(return_embeddings=True),
    )
    # output.embeddings shape: (1, L+2, 1152) — includes BOS/EOS tokens
    # Mean-pool over residue positions, excluding BOS (idx 0) and EOS (idx -1)
    embeddings = output.embeddings[0, 1:-1, :]  # (L, 1152)
    mean_emb = embeddings.mean(dim=0)            # (1152,)
    return mean_emb.cpu()


def process_proteins_on_gpu(
    proteins: list[tuple[str, str]],
    gpu_id: int,
    embedding_dir: str,
):
    """Process a list of proteins on a single GPU.

    Loads the model once per GPU, then iterates through all assigned proteins.
    Skips proteins that already have a .pt file (idempotent).
    """
    device = f"cuda:{gpu_id}"
    print(f"  [GPU {gpu_id}] Loading {ESM_MODEL} ...")
    model = load_model(device)
    print(f"  [GPU {gpu_id}] Model loaded. Processing {len(proteins)} proteins ...")

    done = 0
    skipped = 0
    errors = 0

    for protein_id, sequence in proteins:
        pt_path = os.path.join(embedding_dir, f"{protein_id}.pt")
        if os.path.exists(pt_path):
            skipped += 1
            continue

        try:
            with torch.no_grad():
                embedding = extract_embedding(model, sequence)

            torch.save({"mean_embedding": embedding}, pt_path)
            done += 1

            if done % 100 == 0:
                print(f"  [GPU {gpu_id}] Extracted {done} embeddings ...")
        except Exception as e:
            print(f"  [GPU {gpu_id}] ERROR {protein_id}: {e}")
            errors += 1

    print(
        f"  [GPU {gpu_id}] Done: {done} extracted, {skipped} skipped, {errors} errors"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 (ESM-C): Extract ESM-C 600M protein embeddings"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/dti-scratch/protein-esm2.csv",
        help="Input CSV with protein_id and sequence columns (default: data/dti-scratch/protein-esm2.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/protein-esmc",
        help="Output directory for embedding files (default: data/protein-esmc)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs to use (default: '0')",
    )
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    embedding_dir = os.path.join(args.output_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)

    print("Stage 2 (ESM-C): Extracting ESM-C 600M protein embeddings")
    print("=" * 60)
    print(f"  Model:      {ESM_MODEL}")
    print(f"  Embed dim:  {EMBEDDING_DIM}")
    print(f"  GPUs:       {gpu_ids}")

    # Load protein data (protein_id, sequence columns)
    print(f"\n  Loading {args.input} ...")
    df = pd.read_csv(args.input, usecols=["protein_id", "sequence"])
    proteins = (
        df[["protein_id", "sequence"]]
        .drop_duplicates(subset=["protein_id"])
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

    # Filter to only proteins that need extraction
    proteins_todo = [
        (pid, seq)
        for pid, seq in proteins
        if not os.path.exists(os.path.join(embedding_dir, f"{pid}.pt"))
    ]

    # Split proteins across GPUs (each GPU loads its own model)
    n_gpus = len(gpu_ids)
    chunks = [[] for _ in range(n_gpus)]
    for i, protein in enumerate(proteins_todo):
        chunks[i % n_gpus].append(protein)

    # Pre-download model weights to cache (avoids race condition across threads)
    print(f"\n  Pre-downloading {ESM_MODEL} weights (if not cached) ...")
    load_model("cpu")
    torch.cuda.empty_cache()
    print("  Weights cached.")

    print(f"\n  Distributing {len(proteins_todo)} proteins across {n_gpus} GPU(s) ...")
    for i, gpu_id in enumerate(gpu_ids):
        print(f"    GPU {gpu_id}: {len(chunks[i])} proteins")

    # Process each GPU chunk in a separate thread
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for i, gpu_id in enumerate(gpu_ids):
            if len(chunks[i]) == 0:
                continue
            fut = executor.submit(
                process_proteins_on_gpu, chunks[i], gpu_id, embedding_dir
            )
            futures.append(fut)

        for fut in futures:
            fut.result()

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
