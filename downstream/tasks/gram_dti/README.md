# GRAM-DTI downstream eval

Implements the binary-classification benchmark from the GRAM-DTI paper
(arXiv:2509.21971, ICLR 2026). Swaps the molecule encoder while keeping
ESM-2 protein embeddings fixed.

## Benchmark

| Subset       | Task | CV       | Splits                         |
|--------------|------|----------|--------------------------------|
| Yamanishi 08 | DTI  | 10-fold  | warm / drug_cold / target_cold |
| Hetionet     | DTI  | 10-fold  | warm / drug_cold / target_cold |
| Activation   | MoA  | 5-fold   | warm / drug_cold / target_cold |
| Inhibition   | MoA  | 5-fold   | warm / drug_cold / target_cold |

Negatives sampled 1:10. 360 fold-runs per encoder.

## Run

```bash
# Smoke test — one fold, one method, one subset.
python -m downstream.tasks.gram_dti.eval \
    --encoder minimol --head mlp \
    --subsets activation --methods target_cold --folds 0

# Full sweep for one encoder.
python -m downstream.tasks.gram_dti.eval --encoder minimol --head mlp
python -m downstream.tasks.gram_dti.eval --encoder mole    --head autogluon
python -m downstream.tasks.gram_dti.eval --encoder pairmixer \
    --ckpt models_checkpoints/small-dataset/pairmixer_12M/.../last.ckpt \
    --head mlp
```

Or orchestrate everything via `scripts/gram_dti/run_all.sh`.

## Heads

- `mlp` — sklearn `MLPClassifier` (256-256) inside `StandardScaler`. Fast, reproducible.
- `autogluon` — `autogluon.tabular.TabularPredictor`. Matches the original
  GRAM-DTI repo. Install: `pip install autogluon.tabular`.

## Feature vector

Per (drug, target): `[ mol_encoder(drug) || esm2(target) ]`.

| Encoder   | mol dim | prot dim | total |
|-----------|---------|----------|-------|
| minimol   | 512     | 1280     | 1792  |
| mole      | 256     | 1280     | 1536  |
| pairmixer | ckpt    | 1280     | var   |

## Results

One row per (encoder, subset, method, fold, head) in
`results/downstream/gram_dti.csv`. Columns include
`auroc / auprc / f1 / sensitivity / accuracy` for val + test.

AUPRC is the primary metric (1:10 class imbalance). Reference (paper,
target-cold — hardest split):

| Dataset      | AUROC         | AUPRC         |
|--------------|---------------|---------------|
| Yamanishi 08 | 0.955 ± 0.016 | 0.849 ± 0.031 |
| Hetionet     | 0.921 ± 0.008 | 0.626 ± 0.024 |
| Activation   | 0.834 ± 0.026 | 0.450 ± 0.037 |
| Inhibition   | 0.823 ± 0.003 | 0.464 ± 0.056 |
