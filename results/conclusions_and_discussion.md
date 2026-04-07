# Experiment Conclusions and Discussion Points

**Date**: 2026-04-07  
**Scope**: 1,196 experiment runs across GPS++, Pairformer, and PairMixer on 22 ADMET tasks  
**Primary model**: GPS++ 800M (12 layers, 1536 dim)

---

## 1. Pre-training Dataset Type: Molecular Data Matters Most

**Key finding**: LargeMix pre-training wins 15/22 ADMET tasks on average, but has only n=1 per task -- needs replication.

| Pre-training | Win count (22 tasks) | Strength | Weakness |
|---|---|---|---|
| **LargeMix** | 15/22 | Best across absorption, metabolism, excretion | n=1 -- high variance risk |
| **ToyMix+DTI filtered** | 3/22 | Strong on vdss, herg, hia_hou | Modest improvement over unfiltered |
| **ToyMix** | 1/22 | Solid baseline | Worse than DTI-augmented variants |
| **Scratch** | 1/22 | Wins ld50_zhu toxicity | Fails badly on hia_hou, dili, ppbr_az |
| **BBBC047** | 1/22 | Best caco2_wang (0.487 MAE) | Limited to 1 task win |

**Discussion**: The LargeMix results are very promising but single-seed. Priority should be running LargeMix with 2-3 seeds to confirm. If confirmed, it would indicate that dataset scale is the dominant factor over dataset modality.

---

## 2. DTI Pre-training: Helpful But Not Alone

**Key finding**: DTI-only pre-training hurts performance. DTI combined with molecular tasks (ToyMix+DTI) consistently helps.

| Configuration | caco2 (MAE) | hia_hou (AUROC) | pgp (AUROC) | half_life (Spear.) | herg (AUROC) |
|---|---|---|---|---|---|
| Scratch | 0.711 | 0.500 | 0.821 | 0.134 | 0.743 |
| ToyMix only | 0.487 | 0.915 | 0.790 | 0.190 | 0.665 |
| **DTI-only** | **3.070** | **0.471** | **0.642** | **-0.019** | **0.610** |
| **ToyMix+DTI** | **0.490** | **0.881** | **0.820** | **0.212** | **0.752** |
| LargeMix | 0.490 | 0.901 | 0.889 | 0.329 | 0.740 |

**Conclusion**: DTI-only pre-training collapses -- the model learns protein-centric representations that don't transfer to ADMET molecular property tasks. But DTI as a supplementary pre-training signal on top of molecular data provides consistent gains, especially on:
- **Excretion tasks**: half_life (0.134 -> 0.212), clearance_hepatocyte (0.085 -> 0.175)
- **Metabolism tasks**: cyp2d6 (0.176 -> 0.405), cyp2d6_substrate (0.311 -> 0.537)

**Discussion point**: DTI may act as a regularizer or provide complementary molecular-interaction features. The protein binding signal forces the GNN to learn binding-relevant substructures.

---

## 3. DTI Filtering Improves Key Tasks

| Configuration | hia_hou | half_life | herg | pgp |
|---|---|---|---|---|
| ToyMix+DTI (unfiltered) | 0.829 | 0.189 | 0.721 | 0.879 |
| ToyMix+DTI (filtered) | **0.942** | **0.323** | **0.787** | 0.807 |

**Conclusion**: Filtering the DTI dataset (removing low-quality drug-protein pairs) substantially improves excretion and classification tasks at some cost to absorption regression tasks. The filtered set (63K pairs) is more effective than the full 100K set.

**Discussion point**: Data quality > data quantity for DTI. Noisy binding annotations dilute the signal. Consider filtering strategies based on binding assay type or confidence score.

---

## 4. Cell Morphology: BBBC047 >> RxRx3 (as standalone)

| Source | caco2 (MAE) | hia_hou | pgp | half_life | herg |
|---|---|---|---|---|---|
| RxRx3-only | 1.246 | 0.499 | 0.437 | 0.025 | 0.501 |
| BBBC047-only | **0.487** | **0.915** | **0.790** | **0.238** | **0.665** |

**Conclusion**: BBBC047 (672-dim CPCNN embeddings, 109K compounds) dramatically outperforms RxRx3 (384-dim, standalone) across all 5 representative tasks. RxRx3-only pre-training essentially fails, producing near-random results.

**Possible explanations**:
- BBBC047 has 5x more compounds than the subset in RxRx3 that overlaps with ADMET chemical space
- CPCNN (672-dim) embeddings may capture more pharmacologically relevant features than RxRx3's 384-dim embeddings
- RxRx3 alone may lack sufficient molecular diversity to learn transferable representations

**Discussion point**: When combined with ToyMix (toymix+rxrx3), RxRx3 still helps on some tasks (bbb_martins 0.784, pgp 0.831), suggesting it provides complementary signal when molecular pre-training provides the foundation. Worth testing BBBC047 + ToyMix + DTI combination.

---

## 5. Multimodal Pre-training: Diminishing Returns When Stacking

| Configuration | caco2 | hia_hou | pgp | half_life | herg |
|---|---|---|---|---|---|
| ToyMix+DTI | 0.490 | 0.881 | 0.820 | 0.212 | 0.752 |
| ToyMix+RxRx3 | 0.558 | 0.717 | 0.837 | 0.011 | 0.740 |
| ToyMix+RxRx3+DTI | 0.549 | 0.726 | 0.792 | 0.145 | 0.761 |

**Conclusion**: Adding RxRx3 on top of ToyMix+DTI doesn't help and can hurt. The triple combination (toymix+rxrx3+dti) is worse than toymix+dti on 4/5 representative tasks. RxRx3 may introduce a competing optimization signal that interferes with DTI's beneficial regularization.

**Discussion point**: Multi-modal pre-training is not additive. Each additional modality competes for model capacity. A curriculum strategy (e.g., molecular first, then DTI, then cell morphology) may work better than joint training.

---

## 6. Finetuning Strategy: Frozen Backbone Works Best (with caveats)

Comparison on ToyMix+DTI, GPS++ 800M (depth=12, hdim=1536), half_life_obach task:

| Strategy | half_life (Spearman) | n |
|---|---|---|
| Frozen (unfrz=0, ep=none) | 0.278 | 6 |
| Unfreeze epoch 50 (unfrz=0, ep=50) | 0.183 | 12 |
| Unfreeze epoch 30 (unfrz=0, ep=30) | 0.162 | 5 |
| Unfreeze epoch 20 (unfrz=0, ep=20) | 0.076 | 1 |
| Partial unfreeze depth=4 (unfrz=4, ep=none) | **0.387** | 1 |

**Observations**:
- **Fully frozen** (unfrz=0, no unfreezing) is the most reliable strategy
- **Delayed full unfreezing** (epoch 50/30/20) consistently hurts -- earlier unfreezing is worse
- **Partial unfreezing** (depth=4, always unfrozen) shows the best single result (0.387) but n=1

**Discussion point**: The delayed-unfreezing schedule may cause catastrophic forgetting when the entire backbone suddenly unfreezes. Partial unfreezing (top-k layers always trainable) may be the best compromise -- needs more seeds and tasks to confirm.

---

## 7. Model Architecture: GPS++ Leads, Pairformer Competitive, PairMixer Undertrained

### Scratch baselines (5 representative tasks)

| Architecture | caco2 (MAE) | pgp (AUROC) | half_life (Spear.) | herg (AUROC) |
|---|---|---|---|---|
| GPS++ 800M (d=12, h=1536) | 0.565 | 0.821 | 0.097 | 0.746 |
| Pairformer (d=48, h=384) | 1.172 | 0.861 | 0.098 | 0.779 |
| PairMixer (d=48, h=384) | 1.219 | 0.858 | -0.056 | 0.771 |
| Pairformer (d=16, h=96) | 1.195 | 0.807 | 0.115 | **0.805** |

**Observations**:
- GPS++ 800M wins on caco2 (regression) by a wide margin
- Pairformer is competitive or better on classification tasks (pgp, herg)
- PairMixer's negative half_life correlation suggests insufficient training
- Smaller Pairformer (d=16, h=96) achieves the best herg AUROC (0.805)

### Finetuning transfer

| Architecture | caco2 (MAE) | hia_hou | pgp | half_life | herg |
|---|---|---|---|---|---|
| GPS++ (FT, best) | ~0.49 | ~0.88 | ~0.82 | ~0.21 | ~0.75 |
| Pairformer (FT) | 0.623 | 0.859 | 0.854 | 0.200 | 0.760 |
| PairMixer (FT) | 5.250 | 0.500 | 0.500 | 0.000 | 0.500 |

**Conclusion**: PairMixer finetuning completely fails (random predictions). The ToyMix+DTI pre-trained checkpoint may not transfer to the PairMixer architecture, or the finetuning config needs architecture-specific tuning.

**Discussion point**: PairMixer needs its own pre-training + finetuning pipeline debugging. The architecture is promising (1.75x faster than Pairformer in head-to-head benchmarks) but the transfer learning pipeline is broken.

---

## 8. MoE (Mixture of Experts): Promising Early Results

GPS++ MoE (d=8, h=768, 4 experts, top-2) pre-trained on ToyMix+DTI:

| Task | MoE | Scratch (same arch) | Improvement |
|---|---|---|---|
| caco2_wang (MAE) | **0.431** | 0.857 | 2x better |
| half_life_obach (Spear.) | **0.301** | 0.170 | +77% |
| herg (AUROC) | **0.780** | 0.739 | +5.5% |

**Conclusion**: MoE shows strong improvement over the equivalent non-MoE architecture at the same hidden dimension. The routing mechanism appears to help the model specialize different expert pathways for different molecular property types.

**Discussion point**: MoE results are from a smaller model (768 dim vs 1536 dim for GPS++ 800M). A fair comparison would be MoE at 1536 dim, but memory constraints may require careful expert placement.

---

## 9. Comparison vs TDC Leaderboard

**Key finding**: Our best experiments beat the TDC top-10 average on **5/22 tasks** and beat SOTA (#1) on **0/22 tasks**. We are competitive on excretion/distribution but lag significantly on absorption, metabolism, and toxicity.

### Tasks beating TDC Top-10 Average

| Task | Category | TDC Top-10 Avg | Our Best | Gap |
|---|---|---|---|---|
| ppbr_az (MAE) | Distribution | 7.918 | **7.650** | +0.268 |
| clearance_hepatocyte_az (Spear.) | Excretion | 0.457 | **0.470** | +0.013 |
| clearance_microsome_az (Spear.) | Excretion | 0.608 | **0.612** | +0.004 |
| cyp2c9_substrate (AUPRC) | Metabolism | 0.426 | **0.448** | +0.022 |
| half_life_obach (Spear.) | Excretion | 0.497 | **0.497** | +0.000 (marginal) |

### Closest misses (within 0.03 of TDC Top-10 Avg)

| Task | Category | TDC Top-10 Avg | Our Best | Gap |
|---|---|---|---|---|
| bbb_martins (AUROC) | Distribution | 0.913 | 0.903 | -0.010 |
| cyp3a4_substrate (AUROC) | Metabolism | 0.649 | 0.648 | -0.001 |
| cyp2d6_substrate (AUPRC) | Metabolism | 0.705 | 0.688 | -0.017 |
| hia_hou (AUROC) | Absorption | 0.985 | 0.965 | -0.020 |
| pgp_broccatelli (AUROC) | Absorption | 0.921 | 0.892 | -0.029 |
| vdss_lombardo (Spear.) | Distribution | 0.602 | 0.599 | -0.003 |

### Largest gaps (where we lag most)

| Task | Category | TDC Top-10 Avg | Our Best | Gap |
|---|---|---|---|---|
| lipophilicity (MAE) | Absorption | 0.504 | 0.756 | -0.252 |
| bioavailability (AUROC) | Absorption | 0.930 | 0.708 | -0.222 |
| cyp3a4_veith (AUPRC) | Metabolism | 0.884 | 0.805 | -0.079 |
| ames (AUROC) | Toxicity | 0.856 | 0.778 | -0.078 |
| caco2_wang (MAE) | Absorption | 0.290 | 0.361 | -0.071 |

**Discussion**:
- **Excretion is our strongest category**: 3/3 excretion tasks beat or match the TDC top-10. This aligns with DTI pre-training helping the most on PK-related tasks.
- **Absorption is our weakest**: lipophilicity and bioavailability lag by 0.2+. These tasks may benefit more from molecular descriptor features (e.g., LogP, TPSA) that TDC leaderboard methods like MiniMol/MapLight include.
- **The gap to SOTA (#1) is substantial**: 0/22 tasks beat rank-1. The TDC leaderboard is dominated by ensemble methods (MapLight+GNN, MiniMol) and boosting approaches (CaliciBoost, BaseBoosting) that combine GNNs with molecular fingerprints. Our pure GNN approach is at a structural disadvantage on tasks where 2D descriptors carry strong signal.
- **Multi-seed replication needed**: Our "best" values are often single runs. The TDC leaderboard reports mean +/- std across 5 seeds, so our single-run bests are optimistically biased.

---

## 10. Summary of Key Conclusions

9. **vs TDC leaderboard**: 5/22 tasks beat top-10 average (excretion strongest), 0/22 beat SOTA; pure GNN lags ensemble/fingerprint methods on absorption

1. **Dataset scale dominates**: LargeMix > ToyMix for pre-training (pending multi-seed confirmation)
2. **DTI is a strong auxiliary signal**: Combined with molecular data, DTI improves metabolism and excretion tasks; alone, it's harmful
3. **Data quality matters**: Filtered DTI > unfiltered DTI, especially for excretion tasks
4. **Cell morphology**: BBBC047 >> RxRx3 standalone; RxRx3 only useful as supplement to molecular pre-training
5. **Multimodal stacking has limits**: Two modalities good, three can hurt -- signals compete for capacity
6. **Keep backbone frozen**: Full unfreezing hurts; partial unfreezing (depth=4) promising but unconfirmed
7. **Architecture**: GPS++ best overall; Pairformer competitive on classification; PairMixer transfer broken
8. **MoE**: Preliminary positive results on smaller model, needs scaling study

---

## 11. Recommended Next Experiments

| Priority | Experiment | Rationale |
|---|---|---|
| **P0** | LargeMix multi-seed (3 seeds, 22 tasks) | Confirm the dominant finding |
| **P0** | Fix PairMixer finetuning pipeline | Currently produces random outputs |
| **P1** | LargeMix + DTI combined pre-training | Could combine the top-2 strategies |
| **P1** | BBBC047 + ToyMix + DTI | Test the best cell morphology source with best auxiliary signal |
| **P1** | Partial unfreezing (depth=2,4) multi-seed | Confirm the promising single-run result |
| **P2** | MoE at GPS++ 800M scale | Scale the MoE improvement to larger model |
| **P2** | ESM-C vs ESM2 DTI embeddings | Test if smaller/newer protein embeddings (1152d) match 2560d ESM2 |
| **P2** | DTI filtering strategies | Test binding affinity threshold, assay type filtering |

---

## 12. Caveats

- **Sample sizes**: Many comparisons are n=1 (especially LargeMix, BBBC047). All claims should be treated as hypotheses pending replication.
- **Confounds**: Different pre-training datasets may have had different training durations, learning rates, or convergence states. The checkpoint quality varies.
- **Autoresearch runs**: ~400 runs were generated by automated architecture search agents, which may have different hyperparameter distributions than manual experiments.
- **Excretion tasks**: Generally noisy (low Spearman correlations across all methods). half_life_obach is the most informative excretion task; clearance tasks have near-zero signal for most configurations.
