#!/usr/bin/env python
"""Stage 2 (LPM-24): Extract PubMedBERT embeddings from molecule captions.

Takes the raw LPM-24 parquet from Stage 1 and runs PubMedBERT
(microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) to produce
mean-pooled embeddings (dimension 768) from molecule captions.

Output:
    <output-dir>/lpm24_embeddings.pt:
        {"embeddings": FloatTensor[N, 768],
         "smiles": list[str]}

The process saves checkpoints every --save-every batches. Existing embeddings
are loaded and only remaining rows are processed (idempotent).

Prerequisites:
    pip install transformers torch

Runtime: ~20-30 minutes for 160,560 captions on a single GPU.

Usage:
    python 02_extract_embeddings_pubmedbert.py \\
        --input data/lpm24/lpm24_raw.parquet \\
        --output-dir data/lpm24 \\
        --gpu 0 \\
        --text-field caption
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
EMBEDDING_DIM = 768
MAX_SEQ_LEN = 512


def mean_pool_batch(model, tokenizer, texts, device, max_length=MAX_SEQ_LEN):
    """Compute attention-masked mean pooling for a batch of texts.

    Returns:
        FloatTensor of shape (batch_size, 768).
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc)
    # Mask padding tokens before averaging
    mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
    summed = (out.last_hidden_state * mask).sum(dim=1)  # (B, 768)
    counts = mask.sum(dim=1)                             # (B, 1)
    return (summed / counts).cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 (LPM-24): Extract PubMedBERT embeddings"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/lpm24/lpm24_raw.parquet",
        help="Input parquet from Stage 1 (default: data/lpm24/lpm24_raw.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lpm24",
        help="Output directory (default: data/lpm24)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for PubMedBERT inference (default: 64)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="caption",
        choices=["caption", "properties_str"],
        help="Which text field to embed (default: caption)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N batches (default: 100)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    output_path = os.path.join(args.output_dir, "lpm24_embeddings.pt")

    print("Stage 2 (LPM-24): Extracting PubMedBERT embeddings")
    print("=" * 60)
    print(f"  Model:      {MODEL_ID}")
    print(f"  Embed dim:  {EMBEDDING_DIM}")
    print(f"  Text field: {args.text_field}")
    print(f"  Device:     {device}")
    print(f"  Batch size: {args.batch_size}")

    # Load raw data
    print(f"\n  Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    texts = df[args.text_field].tolist()
    smiles = df["molecule"].tolist()
    print(f"  Total molecules: {len(df):,}")

    # Check for existing checkpoint
    start_idx = 0
    embeddings_list = []
    if os.path.exists(output_path):
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        existing_n = checkpoint["embeddings"].shape[0]
        if existing_n < len(texts):
            print(f"  Resuming from checkpoint: {existing_n:,}/{len(texts):,} done")
            embeddings_list.append(checkpoint["embeddings"].numpy())
            start_idx = existing_n
        elif existing_n == len(texts):
            print(f"  All {len(texts):,} embeddings already extracted. Nothing to do.")
            return
        else:
            print(f"  WARNING: checkpoint has {existing_n} rows but data has {len(texts)}. Re-extracting.")

    # Load model and tokenizer
    print(f"\n  Loading PubMedBERT ...")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print(f"  Model loaded on {device}.")

    # Process batches
    remaining_texts = texts[start_idx:]
    n_batches = (len(remaining_texts) + args.batch_size - 1) // args.batch_size
    print(f"\n  Processing {len(remaining_texts):,} texts in {n_batches} batches ...")

    batch_embeddings = []
    for i in tqdm(range(0, len(remaining_texts), args.batch_size), desc="  Embedding"):
        batch_texts = remaining_texts[i : i + args.batch_size]
        emb = mean_pool_batch(model, tokenizer, batch_texts, device)
        batch_embeddings.append(emb.numpy())

        # Periodic checkpoint
        batch_num = i // args.batch_size + 1
        if batch_num % args.save_every == 0:
            all_so_far = np.concatenate(embeddings_list + batch_embeddings, axis=0)
            torch.save(
                {"embeddings": torch.FloatTensor(all_so_far), "smiles": smiles[:len(all_so_far)]},
                output_path,
            )
            tqdm.write(f"    Checkpoint saved: {len(all_so_far):,} embeddings")

    # Final save
    all_embeddings = np.concatenate(embeddings_list + batch_embeddings, axis=0)
    assert all_embeddings.shape == (len(texts), EMBEDDING_DIM), (
        f"Shape mismatch: {all_embeddings.shape} != ({len(texts)}, {EMBEDDING_DIM})"
    )

    torch.save(
        {"embeddings": torch.FloatTensor(all_embeddings), "smiles": smiles},
        output_path,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Saved: {output_path}  (shape: {all_embeddings.shape}, {size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
