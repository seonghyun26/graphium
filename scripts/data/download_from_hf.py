#!/usr/bin/env python
"""Download all (or selected) archives from the graphium-pretrain-data HF repo
and extract them to the paths specified in its MANIFEST.json.

Usage (run from anywhere; paths in the manifest are relative to --root):
    # Everything
    python scripts/data/download_from_hf.py

    # Just one or more groups
    python scripts/data/download_from_hf.py --groups lpm24
    python scripts/data/download_from_hf.py --groups lpm24,bbbc047

    # Custom destination root (default: current working dir)
    python scripts/data/download_from_hf.py --root /scratch/molrepr

    # Skip extraction (just download tarballs to --cache-dir)
    python scripts/data/download_from_hf.py --no-extract

The repo is public, so no HF token is required. If it ever becomes private,
set HF_TOKEN in the environment or run `huggingface-cli login` first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "hyunnnnnnnn/graphium-pretrain-data"


def fetch_manifest(cache_dir: Path) -> dict:
    fp = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="MANIFEST.json",
        cache_dir=str(cache_dir),
    )
    return json.loads(Path(fp).read_text())


def extract_strip1(tar_path: Path, dest: Path) -> None:
    """Extract tar to ``dest``, stripping the leading ``source_basename/`` dir.

    Mirrors ``tar -xf <archive> --strip-components=1 -C <dest>``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("tar"):
        subprocess.run(
            ["tar", "-xf", str(tar_path), "--strip-components=1", "-C", str(dest)],
            check=True,
        )
        return
    # Pure-python fallback
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue  # the top-level dir entry itself
            member.name = parts[1]
            tf.extract(member, dest)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root dir; manifest extract_to paths are relative to this. Default: CWD.",
    )
    p.add_argument(
        "--groups",
        default=None,
        help="Comma-sep list of groups to download (e.g. 'lpm24,bbbc047'). Default: all.",
    )
    p.add_argument(
        "--archives",
        default=None,
        help="Comma-sep list of explicit archive paths to download (overrides --groups).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface",
        help="HF download cache. Default: ~/.cache/huggingface",
    )
    p.add_argument(
        "--no-extract",
        action="store_true",
        help="Download tarballs only; don't extract.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if destination already has files.",
    )
    args = p.parse_args()

    print(f"Repo: {REPO_ID}")
    print(f"Root: {args.root}")
    print(f"Cache: {args.cache_dir}")
    print()

    manifest = fetch_manifest(args.cache_dir)
    archives = manifest.get("archives", [])

    if args.archives:
        wanted = {a.strip() for a in args.archives.split(",") if a.strip()}
        archives = [a for a in archives if a["archive"] in wanted]
    elif args.groups:
        wanted = {g.strip() for g in args.groups.split(",") if g.strip()}
        archives = [a for a in archives if a["group"] in wanted]

    if not archives:
        sys.exit("ERROR: no archives matched the filter.")

    print(f"Selected {len(archives)} archive(s):")
    for a in archives:
        print(f"  - {a['group']:<10} {a['archive']:<55} -> {a['extract_to']}")
    print()

    args.root.mkdir(parents=True, exist_ok=True)

    for a in archives:
        archive_rel = a["archive"]
        extract_to = args.root / a["extract_to"]
        print(f"==> {archive_rel}")
        tar_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=archive_rel,
                cache_dir=str(args.cache_dir),
            )
        )
        size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"    downloaded: {tar_path} ({size_mb:.1f} MB)")

        if args.no_extract:
            continue

        if extract_to.exists() and any(extract_to.iterdir()) and not args.force:
            print(f"    skip extract: {extract_to} not empty (--force to override)")
            continue

        extract_strip1(tar_path, extract_to)
        print(f"    extracted -> {extract_to}")

    print("\nDone.")


if __name__ == "__main__":
    main()
