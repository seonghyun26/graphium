# MolE Baseline Utilities

Download the official MolE baseline artifacts linked from the upstream README:

```bash
bash scripts/mole/download_mole_baseline.sh
```

Default output: `downloads/mole/` (git-ignored).

The script downloads Zenodo record `10803099` and verifies the published MD5 checksums for:
- `config.yaml`
- `model.pth`

Run the ADMET/Polaris baseline evaluation:

```bash
bash scripts/mole/run_mole_admet.sh 0
```

This evaluation script:
- clones the upstream MolE repo to `downloads/mole/upstream/` on first use
- checks out a pinned upstream commit
- wires the official Zenodo checkpoint into the upstream `ckpt/` layout
- extracts MolE embeddings with a local cache in `datacache/mole_embeddings/`
- fits sklearn linear/MLP heads like the existing MiniMol baseline workflow
