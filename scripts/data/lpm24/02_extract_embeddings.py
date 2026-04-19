#!/usr/bin/env python
"""Stage 2 (LPM-24): Extract text-encoder embeddings from molecule captions.

Generic, model-agnostic extractor. Select the backbone with ``--model``:

    pubmedbert  : microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
                  BERT, ~110M params, 768-d.  Biomedical (PubMed abstracts + PMC full-text).
    galactica   : facebook/galactica-125m
                  OPT (decoder), ~125M params, 768-d.  Scientific LLM trained on 48M
                  papers + SMILES + IUPAC; chemistry-aware.  The only public Galactica
                  variant with hidden_size <= 1024 (1.3B -> 2048-d, 6.7B -> 4096-d).
    biolinkbert : michiyasunaga/BioLinkBERT-large
                  BERT, ~340M params, 1024-d.  PubMed + citation graph (relational).

All three satisfy: single-GPU-friendly and output_dim <= 1024.

Output
------
    <output-dir>/lpm24_embeddings_<model>.pt:
        {
            "embeddings":    FloatTensor[N, D],
            "smiles":        list[str],
            "model_id":      str,
            "embedding_dim": int,
        }

Checkpoints every ``--save-every`` batches; resumes from partial files (idempotent).

Usage
-----
    python 02_extract_embeddings.py --model galactica   --gpu 1
    python 02_extract_embeddings.py --model biolinkbert --gpu 1
    python 02_extract_embeddings.py --model pubmedbert  --gpu 1
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


MODEL_REGISTRY = {
    "pubmedbert": {
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "embedding_dim": 768,
    },
    "galactica": {
        "model_id": "facebook/galactica-125m",
        "embedding_dim": 768,
    },
    "biolinkbert": {
        "model_id": "michiyasunaga/BioLinkBERT-large",
        "embedding_dim": 1024,
    },
}
MAX_SEQ_LEN = 512


def mean_pool_batch(model, tokenizer, texts, device, max_length=MAX_SEQ_LEN):
    """Attention-masked mean pooling over the last hidden state.

    Works for both bidirectional (BERT) and causal (OPT/Galactica) backbones:
    ``AutoModel`` returns the base transformer in both cases, and
    ``last_hidden_state`` is populated identically.
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    # Only forward keys the model accepts: OPT/Galactica rejects token_type_ids.
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    mask = attention_mask.unsqueeze(-1).float()              # (B, T, 1)
    summed = (out.last_hidden_state * mask).sum(dim=1)       # (B, D)
    counts = mask.sum(dim=1).clamp(min=1)                    # (B, 1)
    return (summed / counts).cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 (LPM-24): Extract text-encoder embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Which text encoder to use",
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
        default=1,
        help="GPU ID to use (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
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

    cfg = MODEL_REGISTRY[args.model]
    model_id = cfg["model_id"]
    embedding_dim = cfg["embedding_dim"]
    output_path = os.path.join(args.output_dir, f"lpm24_embeddings_{args.model}.pt")

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print(f"Stage 2 (LPM-24): Extracting {args.model} embeddings")
    print("=" * 60)
    print(f"  Model:      {model_id}")
    print(f"  Embed dim:  {embedding_dim}")
    print(f"  Text field: {args.text_field}")
    print(f"  Device:     {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output:     {output_path}")

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
        existing_dim = checkpoint["embeddings"].shape[1]
        if existing_dim != embedding_dim:
            raise ValueError(
                f"Checkpoint at {output_path} has dim {existing_dim}, "
                f"but model '{args.model}' expects {embedding_dim}. "
                f"Delete the file or choose a different --model."
            )
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
    print(f"\n  Loading {model_id} ...")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    # Galactica's tokenizer ships with none of pad/eos/bos wired up even though
    # the vocab contains '<pad>'/'</s>' and the model config points at them.
    # Cover all three cases (tokenizer attr -> config fallback -> new special token).
    if tokenizer.pad_token is None:
        pad_id = (
            tokenizer.eos_token_id
            or getattr(model.config, "pad_token_id", None)
            or getattr(model.config, "eos_token_id", None)
        )
        if pad_id is not None:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded on {device}  ({n_params / 1e6:.1f}M params).")

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
                {
                    "embeddings": torch.FloatTensor(all_so_far),
                    "smiles": smiles[: len(all_so_far)],
                    "model_id": model_id,
                    "embedding_dim": embedding_dim,
                },
                output_path,
            )
            tqdm.write(f"    Checkpoint saved: {len(all_so_far):,} embeddings")

    # Final save
    all_embeddings = np.concatenate(embeddings_list + batch_embeddings, axis=0)
    assert all_embeddings.shape == (len(texts), embedding_dim), (
        f"Shape mismatch: {all_embeddings.shape} != ({len(texts)}, {embedding_dim})"
    )
    torch.save(
        {
            "embeddings": torch.FloatTensor(all_embeddings),
            "smiles": smiles,
            "model_id": model_id,
            "embedding_dim": embedding_dim,
        },
        output_path,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Saved: {output_path}  (shape: {all_embeddings.shape}, {size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
