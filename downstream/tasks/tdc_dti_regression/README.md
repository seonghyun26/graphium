# TDC DTI regression downstream eval

Sibling of `gram_dti` for TDC's **regression** DTI benchmarks:

| Subset              | Label      | Methods                         |
|---------------------|------------|---------------------------------|
| DAVIS               | pY         | random / cold_target (5 seeds)  |
| KIBA                | kiba_score | random / cold_target (5 seeds)  |
| BindingDB_Kd        | pY         | random / cold_target (5 seeds)  |
| BindingDB_Ki        | pY         | random / cold_target (5 seeds)  |
| BindingDB_IC50      | pY         | random / cold_target (5 seeds)  |
| BindingDB_Patent_DG | pY         | temporal (TDC leaderboard)      |

## Run

```bash
# DAVIS + KIBA regression, MLP head, 5 seeds per method.
python -m downstream.tasks.tdc_dti_regression.eval \
    --encoder minimol --head mlp \
    --subsets DAVIS KIBA --methods random cold_target --seeds 0 1 2 3 4

# TDC leaderboard: BindingDB_Patent_DG temporal split.
python -m downstream.tasks.tdc_dti_regression.eval \
    --encoder pairmixer --ckpt path/to.ckpt \
    --subsets BindingDB_Patent_DG --methods temporal --seeds 0 1 2 3 4
```

## Metrics

`mae / mse / r2_score / pearsonr / spearmanr / concordance_index`
— the last is the DTI-literature standard (DeepDTA / GraphDTA / MolTrans).

Compare PCC (pearsonr) on `BindingDB_Patent_DG / temporal` against the TDC
leaderboard: <https://tdcommons.ai/benchmark/dti_dg_group/bindingdb_patent/>.

Results land in `results/downstream/tdc_dti_regression.csv`.
