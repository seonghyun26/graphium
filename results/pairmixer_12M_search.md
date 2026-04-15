# PairMixer ~12M Architecture + Pair-Init Search

*Generated from `results/experiment_results.csv` by `scripts/pairmixer_search_report.py`.*


## Setup

- Model: `pairmixer_auto` (scratch, seed=0, 100 epochs, GPU 0)

- Tasks (5, one per ADMET category): `caco2_wang`, `bbb_martins`, `cyp2d6_veith`, `clearance_microsome_az`, `ld50_zhu`

- Virtual node: `logsum`; `pre_nn_edges.out_dim=64`; no `torch.compile`.

- Helpfulness = mean of per-task [0,1]-normalised score (higher-is-better, sign-flipped for MAE).


## Phase 1 — Architecture sweep

| cfg | D_s | D_z | depth | params | helpfulness | tasks_ok |
|---|---|---|---|---|---|---|

| **D** | 128 | 256 | 7 | 13.2 M | 0.800 | 5/5 |

| **C** | 128 | 192 | 12 | 12.7 M | 0.668 | 5/5 |

| **E** | 192 | 192 | 10 | 11.3 M | 0.664 | 5/5 |

| **B** | 128 | 160 | 16 | 12.0 M | 0.509 | 5/5 |


### Per-task raw scores (phase 1)

```

task  bbb_martins  caco2_wang  clearance_microsome_az  cyp2d6_veith  ld50_zhu
cfg                                                                          
B          0.7429      1.0703                 -0.0249        0.1440    0.7266
C          0.7343      1.2109                  0.0000        0.1687    0.7266
D          0.7097      1.0625                  0.0000        0.1687    0.7188
E          0.7309      1.1094                  0.0000        0.1687    0.7383
```


## Phase 2 — Pair-Init sweep (on best arch: `D`)

| pair_init | helpfulness | tasks_ok |
|---|---|---|

| **('D', 'struct_path')** —  | 0.602 | 5/5 |

| **('E', 'struct_ef')** —  | 0.538 | 5/5 |

| **('E', 'struct')** —  | 0.444 | 5/5 |

| **('D', 'struct_ef')** —  | 0.333 | 5/5 |

| **('D', 'struct')** —  | 0.322 | 5/5 |

| **('E', 'struct_path')** —  | 0.161 | 5/5 |


### Per-task raw scores (phase 2)

```

task             bbb_martins  caco2_wang  clearance_microsome_az  cyp2d6_veith  ld50_zhu
cfg pi                                                                                  
D   struct            0.7327      1.1016                  0.0000        0.1681    0.7266
    struct_ef         0.7177      0.9688                  0.0000        0.1692    0.7305
    struct_path       0.7532      1.0859                  0.0000        0.1789    0.7305
E   struct            0.7416      1.1094                 -0.0194        0.1687    0.7148
    struct_ef         0.7287      1.0781                  0.1016        0.1672    0.7188
    struct_path       0.7360      1.2109                  0.0000        0.1687    0.7383
```


## Best architecture: `D`

## Phase 3 — Refined arch (≤12M) + expanded features

Features added over baseline (toymix.yaml): atom `hybridization, chirality` (onehot); `mass, electronegativity, vdw-radius, num-ring` (float); edge `conjugated`.

| cfg | D_s | D_z | depth | params | helpfulness | tasks_ok |
|---|---|---|---|---|---|---|
| **F3** | 96 | 224 | 8 | 11.6 M | 0.600 | 5/5 |
| **F1** | 96 | 256 | 6 | 11.5 M | 0.539 | 5/5 |
| **F2** | 128 | 256 | 6 | 11.6 M | 0.364 | 5/5 |
| **F4** | 192 | 192 | 11 | ~11.8 M | 0.252 | 2/5 |

### Per-task raw scores (phase 3)

```
task  bbb_martins  caco2_wang  clearance_microsome_az  cyp2d6_veith  ld50_zhu
cfg                                                                          
F1         0.7351      1.0312                  0.0698        0.1687    0.7148
F2         0.7549      0.9727                  0.0000        0.1687    0.7461
F3         0.7593      1.1641                  0.0000        0.1693    0.7148
F4         0.7354      1.0703                     NaN           NaN       NaN
```
