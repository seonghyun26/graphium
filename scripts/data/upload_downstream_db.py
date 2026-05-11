#!/usr/bin/env python
"""Upload data/db/ (downstream.db + all embeddings) to the HF dataset repo.

Uploads the entire data/db/ directory tree in batches to avoid HF commit
size limits, preserving the modality/dataset/model_vxx/ structure.

Usage (from graphium/):
    # Upload everything
    python scripts/data/upload_downstream_db.py

    # Upload only the SQLite DB
    python scripts/data/upload_downstream_db.py --include "downstream.db"

    # Upload only cell embeddings
    python scripts/data/upload_downstream_db.py --include "cell/**"

    # Make repo public
    python scripts/data/upload_downstream_db.py --public

Requires a HF token with write access:
    huggingface-cli login        # interactive
    export HF_TOKEN=hf_...       # or via env var
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

REPO_ID    = "hyunnnnnnnn/mmf-db"
DB_DIR     = Path(__file__).parent.parent.parent / "data" / "db"
BATCH_SIZE = 150   # files per commit; keeps payloads well under HF limits


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo",    default=REPO_ID, help=f"HF dataset repo ID (default: {REPO_ID})")
    p.add_argument("--dir",     type=Path, default=DB_DIR, help="Local data/db/ directory")
    p.add_argument("--include", default=None, help="Glob pattern to restrict upload (e.g. 'cell/**')")
    p.add_argument("--public",  action="store_true", help="Make repo public (default: private)")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    if not args.dir.exists():
        raise SystemExit(f"Directory not found: {args.dir}")

    # Collect files, apply optional glob filter
    all_files = sorted(f for f in args.dir.rglob("*") if f.is_file())
    if args.include:
        all_files = [f for f in all_files
                     if fnmatch.fnmatch(str(f.relative_to(args.dir)), args.include)]

    if not all_files:
        raise SystemExit("No files matched the filter.")

    api = HfApi()

    print(f"Repo:    {args.repo}")
    print(f"Source:  {args.dir}")
    print(f"Filter:  {args.include or '(all files)'}")
    print(f"Files:   {len(all_files)} total, {BATCH_SIZE} per commit")
    print(f"Access:  {'public' if args.public else 'private'}")
    print()

    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    n_batches = (len(all_files) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(all_files), BATCH_SIZE):
        batch = all_files[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        ops = [
            CommitOperationAdd(
                path_in_repo=str(f.relative_to(args.dir)),
                path_or_fileobj=str(f),
            )
            for f in batch
        ]
        print(f"  Batch {batch_num}/{n_batches}: {len(batch)} files ...")
        api.create_commit(
            repo_id=args.repo,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Upload batch {batch_num}/{n_batches}",
        )

    print(f"\nDone: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
