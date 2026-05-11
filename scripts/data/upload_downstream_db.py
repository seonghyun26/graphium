#!/usr/bin/env python
"""Upload data/db/ (downstream.db + all embeddings) to the HF dataset repo.

Uploads the entire data/db/ directory tree, preserving the
modality/dataset/model_vxx/ structure on the hub.

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
from pathlib import Path

REPO_ID = "hyunnnnnnnn/mmf-db"
DB_DIR  = Path(__file__).parent.parent.parent / "data" / "db"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo",    default=REPO_ID, help=f"HF dataset repo ID (default: {REPO_ID})")
    p.add_argument("--dir",     type=Path, default=DB_DIR, help="Local data/db/ directory to upload")
    p.add_argument("--include", default=None, help="Glob pattern to restrict upload (e.g. 'cell/**')")
    p.add_argument("--public",  action="store_true", help="Make repo public (default: private)")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    if not args.dir.exists():
        raise SystemExit(f"Directory not found: {args.dir}")

    api = HfApi()

    print(f"Repo:    {args.repo}")
    print(f"Source:  {args.dir}")
    print(f"Filter:  {args.include or '(all files)'}")
    print(f"Access:  {'public' if args.public else 'private'}")
    print()

    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(args.dir),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=args.include,
    )

    print(f"\nDone: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
