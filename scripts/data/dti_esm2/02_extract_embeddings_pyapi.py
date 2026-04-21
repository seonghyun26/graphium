#!/usr/bin/env python
"""Python-API variant of stage 2 — extract ESM-2 mean-representation embeddings.

Uses ``esm.pretrained`` directly instead of the ``esm-extract`` CLI, so it
works even when fair-esm's console scripts aren't registered (our
``fair-esm 2.0.0`` install in the ``graphium`` env is missing both the CLI
entry point *and* the ``esm/scripts/`` subpackage).

Output schema is byte-identical to esm-extract: one ``<protein_id>.pt`` per
protein under ``<output_dir>/embedding/`` containing

    {"label": protein_id, "mean_representations": {repr_layer: tensor([D])}}

so downstream consolidation (``02_consolidate_esm2.py``) is unchanged.

Usage
-----
    python scripts/data/dti_esm2/02_extract_embeddings_pyapi.py \\
        --input      ../data/downstream/gram_dti/dtiam-proteins.csv \\
        --output-dir ../data/downstream/gram_dti/protein-esm2 \\
        --gpu        7 \\
        --esm-model  esm2_t33_650M_UR50D \\
        --repr-layer 33
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True,
                   help="CSV with (Target_ID, Target) or (protein_id, sequence) columns.")
    p.add_argument("--output-dir", required=True,
                   help="Parent dir; per-protein .pt files land under <output_dir>/embedding/.")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index (pinned via CUDA_VISIBLE_DEVICES).")
    p.add_argument("--esm-model", default="esm2_t33_650M_UR50D",
                   help="fair-esm pretrained model builder under esm.pretrained.")
    p.add_argument("--repr-layer", type=int, default=33,
                   help="Transformer layer whose mean representation to save.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Proteins per forward pass. Reduce on OOM.")
    p.add_argument("--max-seq-len", type=int, default=1022,
                   help="Truncate sequences above this many residues (ESM-2 context is 1024 incl. BOS/EOS).")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    emb_dir = Path(args.output_dir) / "embedding"
    emb_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "Target_ID" in df.columns and "Target" in df.columns:
        df = df.rename(columns={"Target_ID": "protein_id", "Target": "sequence"})
    if "protein_id" not in df.columns or "sequence" not in df.columns:
        sys.exit(
            f"{args.input}: need columns (Target_ID, Target) or (protein_id, sequence); "
            f"got {list(df.columns)}"
        )
    df["protein_id"] = df["protein_id"].astype(str)
    df = df.dropna(subset=["sequence"]).drop_duplicates("protein_id").reset_index(drop=True)
    proteins: list[tuple[str, str]] = list(zip(df["protein_id"].tolist(), df["sequence"].tolist()))
    print(f"Unique proteins: {len(proteins):,}")

    # Idempotent — skip already-written .pt files.
    pending = [(pid, seq) for pid, seq in proteins if not (emb_dir / f"{pid}.pt").exists()]
    print(f"Already extracted: {len(proteins) - len(pending):,}")
    print(f"Remaining:         {len(pending):,}")
    if not pending:
        return

    # Truncate proteins that exceed ESM-2's context window. Upstream esm-extract
    # does the same — long sequences get silently clipped to the first
    # ``max_seq_len`` residues (informative, not fatal).
    truncated = 0
    for i, (pid, seq) in enumerate(pending):
        if len(seq) > args.max_seq_len:
            pending[i] = (pid, seq[: args.max_seq_len])
            truncated += 1
    if truncated:
        print(f"  WARN: truncated {truncated} proteins to {args.max_seq_len} residues.")

    # Sort by length so each batch has homogeneous padding (faster + less OOM-prone).
    pending.sort(key=lambda x: len(x[1]))

    import esm
    builder = getattr(esm.pretrained, args.esm_model, None)
    if builder is None:
        sys.exit(f"ESM-2 model {args.esm_model!r} not found in esm.pretrained.")
    print(f"Loading {args.esm_model} ...")
    model, alphabet = builder()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("  WARN: torch.cuda.is_available() is False; running on CPU.")
    batch_converter = alphabet.get_batch_converter()
    pad_idx = alphabet.padding_idx

    pbar = tqdm(total=len(pending), desc=f"ESM-2 layer={args.repr_layer}", unit="prot")
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        _labels, _strs, batch_tokens = batch_converter(batch)
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        seq_lens = (batch_tokens != pad_idx).sum(dim=1)

        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[args.repr_layer], return_contacts=False)
        toks = out["representations"][args.repr_layer]   # (B, L, D)

        for i, (pid, _) in enumerate(batch):
            L = int(seq_lens[i])
            # ESM-2 prepends BOS and appends EOS; exclude both from the mean pool.
            mean_vec = toks[i, 1 : L - 1].mean(dim=0).float().cpu()
            torch.save(
                {"label": pid,
                 "mean_representations": {args.repr_layer: mean_vec}},
                emb_dir / f"{pid}.pt",
            )
        pbar.update(len(batch))
    pbar.close()

    n_done = sum(1 for p in proteins if (emb_dir / f"{p[0]}.pt").exists())
    print(f"\nDone. {n_done:,} / {len(proteins):,} proteins have .pt files.")


if __name__ == "__main__":
    main()
