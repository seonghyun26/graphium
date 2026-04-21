# Cell bioactivity downstream eval

Port of the 29-assay ChEMBL / JUMP Cell Painting benchmark from
[Fredinh et al. 2024](https://www.nature.com/articles/s41467-024-47171-1)
(upstream repo: [cfredinh/bioactive](https://github.com/cfredinh/bioactive)).

## Benchmark

- **Labels**: 29 ChEMBL assays, binary active/inactive. NaN entries are masked
  out of the loss (focal BCE with γ=2).
- **Splits**: Butina-clustered 6-fold CV on ECFP4 (r=2, 1024 bits) with
  rotation — test={k}, val={(k+1)%6}, train=rest. Matches upstream.
- **Metric**: macro-AUROC and macro-AUPRC, averaged across the assays that have
  at least two labeled compounds and both classes in the test fold.

## Encoders

| Encoder   | Dim  | Notes                                                      |
|-----------|------|------------------------------------------------------------|
| ecfp      | 1024 | Morgan r=2, the paper's chemistry baseline                 |
| minimol   | 512  | Pretrained Minimol encoder (`pip install minimol`)         |
| mole      | 256  | MolE gin_concat (Zenodo checkpoint, auto-fetched)          |
| pairmixer | var  | Any graphium checkpoint via `--ckpt`                       |
| cpcnn     | 672  | Cell Painting CPCNN embedding (the paper's cell-only baseline) |

## Run

```bash
# 6-fold CV per encoder, writing one row per fold to results/downstream/bioactivity.csv
python -m downstream.tasks.bioactivity.eval --encoder ecfp    --cv
python -m downstream.tasks.bioactivity.eval --encoder minimol --cv
python -m downstream.tasks.bioactivity.eval --encoder mole    --cv
python -m downstream.tasks.bioactivity.eval --encoder cpcnn   --cv \
    --cpcnn-csv /path/to/jump_cpcnn_smiles_embeddings.csv
python -m downstream.tasks.bioactivity.eval --encoder pairmixer --cv \
    --ckpt models_checkpoints/.../*.ckpt

# Or orchestrate via:
bash scripts/bioactivity/run_all.sh [pairmixer_ckpt]
```

## Head

3-layer MLP (paper default `hidden=512 / num_hidden=3 / dropout=0.5`).
Loss is `FocalBCEMaskedLoss` (γ=2). Optimizer auto-resolves:

- **ECFP** → SGD + lr=2.0 (paper default; works for sparse binary inputs)
- **Dense encoders** → Adam + lr=1e-3 (SGD+2.0 explodes on continuous inputs)

Override with `--optimizer` and `--lr` as needed.

## Data

`data/downstream/bioactivity/cell_bioactivity.csv` — columns
`smiles, assay_<id>..., fold`. 0/1/NaN labels (NaN = unknown). Regenerate
with:

```bash
bash scripts/bioactivity/prepare_data.sh /path/to/chembl_33.db
```
