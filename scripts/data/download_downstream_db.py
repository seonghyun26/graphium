#!/usr/bin/env python
"""Download downstream.db from the HF dataset repo.

Usage (from graphium/):
    python scripts/data/download_downstream_db.py
    python scripts/data/download_downstream_db.py --dest /custom/path/downstream.db

The repo is private; requires a HF token with read access:
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
    p.add_argument("--repo", default=REPO_ID, help=f"HF dataset repo ID (default: {REPO_ID})")
    p.add_argument("--dest", type=Path, default=DB_PATH, help="Destination path for downstream.db")
    args = p.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Repo: {args.repo}")
    print(f"Dest: {args.dest}")
    print()

    path = hf_hub_download(
        repo_id=args.repo,
        repo_type="dataset",
        filename="downstream.db",
        local_dir=str(args.dest.parent),
    )

    # hf_hub_download may put it in a cache subdir; rename to exact dest if needed
    downloaded = Path(path)
    if downloaded.resolve() != args.dest.resolve():
        downloaded.rename(args.dest)

    size_mb = args.dest.stat().st_size / 1e6
    print(f"Done: {args.dest}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
