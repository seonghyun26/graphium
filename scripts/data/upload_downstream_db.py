#!/usr/bin/env python
"""Upload data/db/downstream.db to the HF dataset repo.

Usage (from graphium/):
    python scripts/data/upload_downstream_db.py
    python scripts/data/upload_downstream_db.py --public
    python scripts/data/upload_downstream_db.py --repo hyunnnnnnnn/mmf-db

Requires a HF token with write access:
    huggingface-cli login        # interactive
    export HF_TOKEN=hf_...       # or via env var
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID  = "hyunnnnnnnn/mmf-db"
DB_PATH  = Path(__file__).parent.parent.parent / "data" / "db" / "downstream.db"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo",    default=REPO_ID, help=f"HF dataset repo ID (default: {REPO_ID})")
    p.add_argument("--db",      type=Path, default=DB_PATH, help="Path to downstream.db")
    p.add_argument("--public",  action="store_true", help="Make repo public (default: private)")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}\nRun: python data/build_downstream_db.py")

    size_mb = args.db.stat().st_size / 1e6
    api = HfApi()

    print(f"Repo:   {args.repo}")
    print(f"File:   {args.db}  ({size_mb:.1f} MB)")
    print(f"Access: {'public' if args.public else 'private'}")
    print()

    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    api.upload_file(
        path_or_fileobj=str(args.db),
        path_in_repo="downstream.db",
        repo_id=args.repo,
        repo_type="dataset",
    )

    print(f"Done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
