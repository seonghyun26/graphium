#!/usr/bin/env python
"""Download data/db/ (downstream.db + embeddings) from the HF dataset repo.

Downloads the entire repo into data/db/, preserving the
modality/dataset/model_vxx/ structure.

Usage (from graphium/):
    # Download everything
    python scripts/data/download_downstream_db.py

    # Download only the SQLite DB
    python scripts/data/download_downstream_db.py --include "downstream.db"

    # Download only cell embeddings
    python scripts/data/download_downstream_db.py --include "cell/**"

    # Download only protein esmc_v3
    python scripts/data/download_downstream_db.py --include "protein/dti_tdc/esmc_v3/**"

The repo is private; requires a HF token with read access:
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
    p.add_argument("--dest",    type=Path, default=DB_DIR, help="Local destination directory (default: data/db/)")
    p.add_argument("--include", default=None, help="Glob pattern to restrict download (e.g. 'cell/**')")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"Repo:   {args.repo}")
    print(f"Dest:   {args.dest}")
    print(f"Filter: {args.include or '(all files)'}")
    print()

    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(args.dest),
        allow_patterns=args.include,
    )

    print(f"\nDone: {args.dest}")


if __name__ == "__main__":
    main()
