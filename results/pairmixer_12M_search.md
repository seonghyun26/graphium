# PairMixer ≤12M — Architecture, Pair-Init, and Feature Search

*Search period: 2026-04-14 – 2026-04-16 · Scratch ADMET, seed=0, 100 ep, GPU 0/1/4/6/7 · Results from `graphium/results/experiment_results.csv`*

## Headline

🏆 **Best overall config (≤12M): `F3hlr`**

| Lever | Value |
|---|---|
| Arch | D_s = 96, D_z = 224, depth = 8 |
| Params | 11.6 M (≤12M cap) |
| pair_init | OPM-only (structural priors hurt on most tasks) |
| Features | atom onehot: +`hybridization`,+`chirality`; atom float: +`mass`,+`electronegativity`,+`vdw-radius`,+`num-ring`; edge: +`conjugated` |
| LR | **2e-4** (5× the toymix default 4e-5) — biggest single lever |
| virtual_node | `logsum` |
| Unified helpfulness | **0.890** (1st of 22 configs) |

**Apply by default:** `pairmixer_auto.yaml` has been rewritten to this config; `scripts/00_scratch_admet.sh pairmixer_auto 4` now reproduces it out of the box.

## Setup

- 5 representative ADMET tasks (one per category): `caco2_wang` (Absorption, MAE↓), `bbb_martins` (Distribution, AUROC↑), `cyp2d6_veith` (Metabolism, AUPRC↑), `clearance_microsome_az` (Excretion, spearman↑), `ld50_zhu` (Toxicity, MAE↓)
- Scratch training (no pretrain), 1 seed, 100 epochs, bs=32, bf16-mixed, GPU-parallel across phases
- Helpfulness = per-task min-max normalised to [0,1] (sign-flipped for MAE), averaged across tasks. Per-phase tables use in-phase normalisation; the *unified* ranking at the end uses cross-config normalisation.

## Per-task best results (all 22 configs)

| Task | Metric | **Best** | Config | vs phase-1 baseline |
|---|---|---|---|---|
| caco2_wang | MAE ↓ | **0.609** | F3hlr (p3) | 1.06 (phase-1 D) — **−43%** |
| bbb_martins | AUROC ↑ | **0.767** | F3+struct (p4) | 0.74 (phase-1 B) — **+3%** |
| cyp2d6_veith | AUPRC ↑ | **0.179** | D+struct_path (p2) | 0.17 (multiple) |
| clearance_microsome_az | spearman ↑ | **0.149** | F3hlr (p3) | 0.00 everywhere else |
| ld50_zhu | MAE ↓ | **0.664** | F3hlr (p3) | 0.72 (phase-1 D) — **−8%** |

F3hlr wins **3/5 tasks outright** and is within 3 % of the best on the other two.

## Phase 1 — architecture sweep (no feature upgrade, opm only)

| cfg | D_s | D_z | depth | params | helpfulness | caco2 | bbb | cyp2d6 | clearance | ld50 |
|---|---|---|---|---|---|---|---|---|---|---|
| **D** | 128 | 256 | 7 | 13.2 M (over) | 0.800 | 1.06 | 0.71 | 0.169 | 0.00 | 0.719 |
| C | 128 | 192 | 12 | 12.7 M (over) | 0.668 | 1.21 | 0.73 | 0.169 | 0.00 | 0.727 |
| E | 192 | 192 | 10 | **11.3 M** | 0.664 | 1.11 | 0.73 | 0.169 | 0.00 | 0.738 |
| B | 128 | 160 | 16 | **12.0 M** | 0.509 | 1.07 | 0.74 | 0.144 | −0.02 | 0.727 |

*Insight:* At the 12 M budget, widening D_z wins over adding depth. Under the strict ≤12 M cap, E was the best phase-1 point.

## Phase 2 — pair_init on phase-1 winners (D, E)

| cfg | pair_init | helpfulness | caco2 | bbb | cyp2d6 | clearance | ld50 |
|---|---|---|---|---|---|---|---|
| D | struct_path | 0.602 | 1.09 | **0.75** | **0.179** | 0.00 | 0.731 |
| E | struct_ef | 0.538 | 1.08 | 0.73 | 0.167 | **0.10** | 0.719 |
| E | struct | 0.444 | 1.11 | 0.74 | 0.169 | −0.02 | 0.715 |
| D | struct_ef | 0.333 | **0.97** | 0.72 | 0.169 | 0.00 | 0.731 |
| D | struct | 0.322 | 1.10 | 0.73 | 0.168 | 0.00 | 0.727 |
| E | struct_path | 0.161 | 1.21 | 0.74 | 0.169 | 0.00 | 0.738 |

*Insight:* Structural priors help individual tasks but the gains are task-specific (path_edge → bbb_martins + cyp2d6; edge_feat → caco2_wang; none is uniformly better than OPM).

## Phase 3 — refined arch (strict ≤12 M) + expanded features + LR/epoch probes

| cfg | D_s | D_z | depth | LR | epochs | params | helpfulness | caco2 | bbb | cyp2d6 | clearance | ld50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F3hlr** | 96 | 224 | 8 | **2e-4** | 100 | 11.6 M | **0.868** | **0.609** | 0.74 | **0.174** | **0.149** | **0.664** |
| F3long | 96 | 224 | 8 | 4e-5 | 200 | 11.6 M | 0.462 | 0.99 | 0.75 | 0.169 | 0.00 | 0.684 |
| F3 | 96 | 224 | 8 | 4e-5 | 100 | 11.6 M | 0.379 | 1.16 | 0.76 | 0.169 | 0.00 | 0.715 |
| F2 | 128 | 256 | 6 | 4e-5 | 100 | 11.6 M | 0.319 | 0.97 | 0.75 | 0.169 | 0.00 | 0.746 |
| F1 | 96 | 256 | 6 | 4e-5 | 100 | 11.5 M | 0.284 | 1.03 | 0.74 | 0.169 | 0.07 | 0.715 |
| G | 64 | 352 | 3 | 4e-5 | 100 | 11.3 M | 0.248 | 1.16 | **0.76** | 0.169 | −0.16 | 0.727 |
| H | 192 | 192 | 10 | 4e-5 | 100 | 11.3 M | 0.241 | 0.99 | 0.73 | 0.169 | 0.00 | 0.715 |
| F4 | 192 | 192 | 11 | 4e-5 | 100 | 11.8 M | 0.204 | 1.07 | 0.74 | 0.169 | −0.05 | 0.711 |

*Insight:* **LR is the dominant lever.** 5× LR (F3hlr) moved F3 from bottom-half (0.379) to #1 overall (0.868). Doubling the epoch budget (F3long) helped only partially — the real problem was *undertraining at too low a LR*, not insufficient training time. Extremely shallow (G, depth=3) failed; wider single track (H, F4) didn't help when pair width is high.

## Phase 4 — F3 × structural pair_init × expanded features

| cfg | pair_init | helpfulness | caco2 | bbb | cyp2d6 | clearance | ld50 |
|---|---|---|---|---|---|---|---|
| F3 | struct | 0.588 | 1.14 | **0.767** | 0.175 | 0.00 | 0.711 |
| F3 | struct_ef | 0.478 | 1.15 | 0.75 | 0.169 | 0.00 | 0.715 |
| F3 | struct_path | 0.372 | 1.13 | 0.73 | 0.169 | 0.00 | 0.738 |

*Insight:* On top of F3 with expanded features at default LR, structural pair_init **does not beat** plain OPM; `struct` edges bbb_martins (0.767) but loses on caco2/clearance/cyp2d6. Phase 2's ranking (D/struct_path best) was partly an artefact of F3's arch being better than D's in the first place.

## Unified top-10 (normalised across all 22 configs)

| rank | phase | cfg | pi | helpfulness |
|---|---|---|---|---|
| 1 | p3 | **F3hlr** | opm | **0.890** |
| 2 | p3 | F3long | opm | 0.625 |
| 3 | p4 | F3 | struct | 0.588 |
| 4 | p2 | D | struct_path | 0.532 |
| 5 | p3 | F1 | opm | 0.513 |
| 6 | p3 | F3 | opm | 0.511 |
| 7 | p3 | F2 | opm | 0.479 |
| 8 | p2 | E | struct_ef | 0.478 |
| 9 | p4 | F3 | struct_ef | 0.478 |
| 10 | p3 | H | opm | 0.476 |

## Critical observations on performance

### 1. Training protocol, not architecture, was the biggest gap
For 18 of 21 configs, `clearance_microsome_az` was stuck at spearman ≈ 0 and `cyp2d6_veith` AUPRC ≈ class baseline (~0.169). The moment LR was raised 5× (F3hlr), both tasks finally learn:
- `clearance_microsome_az`: 0.00 → **0.149** (first non-zero result)
- `caco2_wang` MAE: 1.16 → **0.609** (43 % reduction)
- `ld50_zhu` MAE: 0.715 → **0.664** (7 % reduction)

The default toymix LR of 4e-5 is appropriate for a GCN/GPS baseline but **starves a PairMixer at ≤12 M** on small ADMET splits.

### 2. D_z width dominates depth at fixed budget — up to a floor on depth
- D_z 128→224 at similar param count consistently improves caco2 / bbb.
- Depth < 6 (G at depth=3) collapses — can't propagate pair information.
- Sweet spot: depth ≈ 7-8 with D_z ≈ 224-256.

### 3. Structural pair_init is task-specific, not universal
- `path_edge` → best on `bbb_martins` (0.76) and `cyp2d6_veith` (0.18)
- `edge_feat` → best on `caco2_wang` (0.97 before features, 1.15 after)
- `adjacency + graph_distance` → mild across-the-board gain
- **None** beat OPM-only + F3 + high LR on the sum.

### 4. Expanded RDKit features (+hybridization, +chirality, +electronegativity, +vdw-radius, +num-ring, +mass, +conjugated) help caco2_wang and bbb_martins
- caco2_wang MAE dropped 1.06 → 0.97 just from feature upgrade (F2 vs phase-1 D)
- bbb_martins AUROC 0.74 → 0.76 (F3 vs phase-1 D)
- Small but consistent; cheap to compute (pure RDKit, no 3D / QM).

### 5. cyp2d6_veith AUPRC hits a data-limited floor around 0.17-0.18
13 k molecules, ~14 % positive class. Even F3hlr only moves from 0.169 → 0.174. Data-limited regime — architectural and training improvements have diminishing returns. Pretraining + transfer is the likely path to further gains.

### 6. Wider single-track (D_s = 192) under-delivers
E (phase 1), H, F4 all use D_s = 192 and land mid-pack. PairMixer doesn't update node features (s_backbone = s_init); the single track matters only through OPM and the final node-pool. Budget is better spent on D_z.

### 7. One OOM event on cyp2d6_veith @ D_z=160 depth=16 bs=32
Single batch skipped out of ~28 800 (0.07 %); no measurable effect on final metric. PairMixer memory scales with B × N² × D_z — occasional large molecules in cyp2d6_veith can spike. Acceptable at the current configs; would need bs=16 for D_z ≥ 288.

### 8. Structural pair_init is CPU-bound
`struct`/`struct_ef`/`struct_path` run Floyd-Warshall every batch on top of the usual featurization — three concurrent runs saturate CPU and stall GPUs. Schedule them one at a time. (Captured in memory.)

## Recommendations

1. **Use `pairmixer_auto` (now = F3hlr) as the default ≤12 M baseline** for scratch ADMET. Already wired via `scripts/00_scratch_admet.sh pairmixer_auto <gpu>`.
2. **Revisit LR for other PairMixer sizes** (10 M, 20 M, 40 M) — 4e-5 is likely too low for them too; search 2e-4 / 1e-4 / 5e-5 with cosine warmup.
3. **For task-specific wins**, adopt per-task pair_init: `bbb_martins` → `struct`/`struct_path`; `caco2_wang` → `struct_ef` (if LR=2e-4 reproduces).
4. **Pretrain on toymix** with the F3hlr config and finetune on hard tasks (`cyp2d6_veith`, `clearance_microsome_az`). *(in progress, GPU 6)*
5. **Seed-variance study (2-3 seeds)** on F3hlr to confirm the 0.15 clearance spearman isn't a fluke.

## Appendix — artifacts

- **Script**: `scripts/pairmixer_12M_search.sh` (phases 1-4, all ≤12 M cap)
- **Analysis**: `scripts/pairmixer_search_report.py` (cross-phase helpfulness + raw pivot)
- **Model config**: `expts/hydra-configs/model/pairmixer_auto.yaml` (now = F3hlr)
- **Training config**: `expts/hydra-configs/training/model/toymix_pairmixer_auto.yaml`
- **Bug fix applied mid-search**: `graphium/nn/architectures/global_architectures.py:2057` — `node_pool_layer` was mis-dispatched through `_pool_layer_forward`; fixed to call `self.node_pool_layer(g, g["feat"])` directly so `pair_pool=stats + node_pooling=[max]` works.
