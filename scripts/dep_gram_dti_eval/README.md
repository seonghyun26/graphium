# GRAM-DTI binary-classification eval

Standalone (non-Hydra, non-graphium-train) eval suite for the 4 DTIAM datasets
benchmarked by GRAM-DTI (ICLR 2026):

| Dataset      | Task | Drugs   | Targets | Positives | CV     |
|--------------|------|---------|---------|-----------|--------|
| Yamanishi 08 | DTI  | 791     | 989     | 5,127     | 10-fold|
| Hetionet     | DTI  | 1,384   | 5,763   | 49,942    | 10-fold|
| Activation   | MoA  | 1,426   | 281     | 1,913     | 5-fold |
| Inhibition   | MoA  | 14,049  | 1,088   | 21,055    | 5-fold |

For every (subset, method, fold) the eval script extracts molecule embeddings
with one encoder (PairMixer / Minimol / MolE), concatenates with the
corresponding ESM-C 1152-d protein embedding, fits a sklearn head
(`logreg` default; `mlp` optional) on train, and reports AUROC / **AUPRC**
(primary) / F1 / Sensitivity / Accuracy on the held-out test fold.

Protocol matches GRAM-DTI exactly: 10/10/5/5 K-fold × {warm, drug_cold,
target_cold} × 1:10 negative sampling = **360 fold-runs per encoder**. See
`memory/project_gram_dti_benchmark.md` for the rationale.

## One-time data prep

```bash
# 1. Clone the DTIAM repo (CSVs only — no LFS).
git clone --depth 1 https://github.com/CSUBioGroup/DTIAM /tmp/DTIAM

# 2. Collect unique protein sequences (filtered to those actually used).
python scripts/data/dti_classif_eval/01_collect_dtiam_proteins.py \
    --dtiam-root /tmp/DTIAM \
    --output    data/dti-classif-eval/dtiam-proteins.csv

# 3. Extract ESM-C 600M embeddings (~25-30 min on 4 GPUs).
/home/shpark/miniforge3/envs/esmc/bin/python \
    scripts/data/dti_esmc/02_extract_embeddings_esmc.py \
    --input      data/dti-classif-eval/dtiam-proteins.csv \
    --output-dir data/dti-classif-eval/protein-esmc \
    --gpus 0,1,2,3

# 4. Consolidate per-protein .pt files into a single parquet.
python scripts/data/dti_esmc/03_consolidate_embeddings_esmc.py \
    --embedding-dir data/dti-classif-eval/protein-esmc/embedding \
    --protein-csv   data/dti-classif-eval/dtiam-proteins.csv \
    --output-dir    data/dti-classif-eval

# 5. Build per-subset parquet + 90 split .pt files.
python scripts/data/dti_classif_eval/02_prepare_dti_classif.py \
    --dtiam-root /tmp/DTIAM \
    --protein-emb data/dti-classif-eval/protein-esmc.parquet
```

## Run the eval

```bash
# Smoke test — one fold, one method, one subset.
python scripts/gram_dti_eval/eval_minimol.py \
    --subsets activation --methods target_cold --folds 0

# Single-encoder sweep (e.g. minimol over everything).
python scripts/gram_dti_eval/eval_minimol.py

# Full sweep across all 3 encoders.
bash scripts/gram_dti_eval/run_all.sh /path/to/pairmixer.ckpt
```

Per-fold rows append to `results/gram_dti_eval/<encoder>_results.csv` with
columns `(encoder, head, subset, method, fold, n_train, n_val, n_test,
val_*, test_*, ...)` so the dashboard notebook can ingest them
the same way it ingests `experiment_results.csv`.

## Files

| Path                  | Purpose                                                                |
|-----------------------|------------------------------------------------------------------------|
| `common.py`           | Shared: data load, per-SMILES feature cache, head training, metrics, sweep loop |
| `eval_minimol.py`     | Minimol (512-d) + ESM-C → 1664-d                                       |
| `eval_mole.py`        | MolE gin_concat (256-d) + ESM-C → 1408-d                               |
| `eval_pairmixer.py`   | PairMixer (z_mol_dim from ckpt) + ESM-C; reuses featurization from `scripts/pairmixer/pairmixer_dti_eval.py` |
| `run_all.sh`          | Orchestrator; iterates encoders × subsets × methods × folds           |

## Reference numbers (GRAM-DTI, target cold start — hardest split)

| Dataset      | AUROC          | AUPRC          |
|--------------|----------------|----------------|
| Yamanishi 08 | 0.955 ± 0.016  | 0.849 ± 0.031  |
| Hetionet     | 0.921 ± 0.008  | 0.626 ± 0.024  |
| Activation   | 0.834 ± 0.026  | 0.450 ± 0.037  |
| Inhibition   | 0.823 ± 0.003  | 0.464 ± 0.056  |
