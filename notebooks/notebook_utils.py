"""Shared constants and utilities for analysis notebooks.

Centralizes ADMET task metadata, model registry, dataset parsing,
and results loading so that notebooks stay DRY.
"""

import re
import json
import statistics
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_CSV = Path(__file__).parent / '../results/experiment_results.csv'
TDC_TOP10_JSON = Path(__file__).parent / '../results/tdc_leaderboard_top10.json'

# ═══════════════════════════════════════════════════════════════════════════════
# ADMET task metadata
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical task ordering: Absorption -> Distribution -> Metabolism -> Excretion -> Toxicity
ADMET_TASK_ORDER = [
    # Absorption
    'caco2_wang', 'hia_hou', 'pgp_broccatelli', 'bioavailability_ma',
    'lipophilicity_astrazeneca', 'solubility_aqsoldb',
    # Distribution
    'bbb_martins', 'ppbr_az', 'vdss_lombardo',
    # Metabolism
    'cyp2d6_veith', 'cyp3a4_veith', 'cyp2c9_veith',
    'cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels',
    # Excretion
    'half_life_obach', 'clearance_hepatocyte_az', 'clearance_microsome_az',
    # Toxicity
    'ld50_zhu', 'herg', 'ames', 'dili',
]

TASK_CATEGORY = {}
for _t in ADMET_TASK_ORDER[:6]:   TASK_CATEGORY[_t] = 'Absorption'
for _t in ADMET_TASK_ORDER[6:9]:  TASK_CATEGORY[_t] = 'Distribution'
for _t in ADMET_TASK_ORDER[9:15]: TASK_CATEGORY[_t] = 'Metabolism'
for _t in ADMET_TASK_ORDER[15:18]: TASK_CATEGORY[_t] = 'Excretion'
for _t in ADMET_TASK_ORDER[18:]:  TASK_CATEGORY[_t] = 'Toxicity'

ADMET_CATEGORIES = ['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']

# Primary metric and direction per task
TASK_METRICS = {
    'caco2_wang': ('mae', 'lower'),
    'hia_hou': ('auroc', 'higher'),
    'pgp_broccatelli': ('auroc', 'higher'),
    'bioavailability_ma': ('auroc', 'higher'),
    'lipophilicity_astrazeneca': ('mae', 'lower'),
    'solubility_aqsoldb': ('mae', 'lower'),
    'bbb_martins': ('auroc', 'higher'),
    'ppbr_az': ('mae', 'lower'),
    'vdss_lombardo': ('spearman', 'higher'),
    'cyp2d6_veith': ('auprc', 'higher'),
    'cyp3a4_veith': ('auprc', 'higher'),
    'cyp2c9_veith': ('auprc', 'higher'),
    'cyp2c9_substrate_carbonmangels': ('auprc', 'higher'),
    'cyp2d6_substrate_carbonmangels': ('auprc', 'higher'),
    'cyp3a4_substrate_carbonmangels': ('auroc', 'higher'),
    'half_life_obach': ('spearman', 'higher'),
    'clearance_hepatocyte_az': ('spearman', 'higher'),
    'clearance_microsome_az': ('spearman', 'higher'),
    'ld50_zhu': ('mae', 'lower'),
    'herg': ('auroc', 'higher'),
    'ames': ('auroc', 'higher'),
    'dili': ('auroc', 'higher'),
}

# TDC Leaderboard SOTA (rank #1): (mean, std, method_name)
# Source: https://tdcommons.ai/benchmark/admet_group/
# Last updated: 2026-04
# Values marked 'MolGPS(paper)' are taken from arXiv:2404.11568 Table 2 where
# MolGPS reports better-than-leaderboard numbers. Std is not reported in that
# paper, so it is set to 0.0. MolGPS code is not publicly released.
TDC_SOTA = {
    'caco2_wang':                       (0.256, 0.006, 'CaliciBoost'),
    'hia_hou':                          (0.993, 0.005, 'MiniMol'),
    'pgp_broccatelli':                  (0.948, 0.000, 'MolGPS(paper)'),
    'bioavailability_ma':               (0.938, 0.002, 'MapLight + GNN'),
    'lipophilicity_astrazeneca':        (0.386, 0.000, 'MolGPS(paper)'),
    'solubility_aqsoldb':               (0.679, 0.000, 'MolGPS(paper)'),
    'bbb_martins':                      (0.941, 0.000, 'MolGPS(paper)'),
    'ppbr_az':                          (6.464, 0.000, 'MolGPS(paper)'),
    'vdss_lombardo':                    (0.713, 0.007, 'MapLight + GNN'),
    'cyp2c9_veith':                     (0.859, 0.001, 'MapLight + GNN'),
    'cyp2d6_veith':                     (0.790, 0.001, 'MapLight + GNN'),
    'cyp3a4_veith':                     (0.916, 0.000, 'MapLight + GNN'),
    'cyp2c9_substrate_carbonmangels':   (0.474, 0.025, 'MiniMol'),
    'cyp2d6_substrate_carbonmangels':   (0.737, 0.024, 'KPGT'),
    'cyp3a4_substrate_carbonmangels':   (0.730, 0.023, 'MolGPS(paper)'),
    'half_life_obach':                  (0.631, 0.000, 'MolGPS(paper)'),
    'clearance_hepatocyte_az':          (0.570, 0.000, 'MolGPS(paper)'),
    'clearance_microsome_az':           (0.633, 0.000, 'MolGPS(paper)'),
    'ld50_zhu':                         (0.552, 0.009, 'BaseBoosting'),
    'herg':                             (0.880, 0.002, 'MapLight + GNN'),
    'ames':                             (0.871, 0.002, 'ZairaChem'),
    'dili':                             (0.956, 0.006, 'MiniMol'),
}

# MolGPS (3B) per-task ADMET results from the paper
# "On the Scalability of GNNs for Molecular Graphs" (arXiv:2404.11568, Table 2).
# Std not reported in the paper, so set to 0.
MOLGPS = {
    'lipophilicity_astrazeneca':        (0.386, 0.0),
    'caco2_wang':                       (0.292, 0.0),
    'ld50_zhu':                         (0.557, 0.0),
    'solubility_aqsoldb':               (0.679, 0.0),
    'ppbr_az':                          (6.464, 0.0),
    'bbb_martins':                      (0.941, 0.0),
    'hia_hou':                          (0.980, 0.0),
    'pgp_broccatelli':                  (0.948, 0.0),
    'bioavailability_ma':               (0.701, 0.0),
    'cyp3a4_substrate_carbonmangels':   (0.680, 0.0),
    'ames':                             (0.857, 0.0),
    'herg':                             (0.864, 0.0),
    'dili':                             (0.942, 0.0),
    'vdss_lombardo':                    (0.649, 0.0),
    'half_life_obach':                  (0.631, 0.0),
    'clearance_microsome_az':           (0.633, 0.0),
    'clearance_hepatocyte_az':          (0.570, 0.0),
    'cyp2d6_substrate_carbonmangels':   (0.713, 0.0),
    'cyp2c9_substrate_carbonmangels':   (0.464, 0.0),
    'cyp2d6_veith':                     (0.750, 0.0),
    'cyp3a4_veith':                     (0.900, 0.0),
    'cyp2c9_veith':                     (0.838, 0.0),
}

# MolE per-task ADMET results from the published Nature Communications 2024 paper
# "MolE: a molecular foundation model for drug discovery" (doi:10.1038/s41467-024-53751-y),
# Table 1. These supersede the arXiv v1 (2022 leaderboard snapshot) numbers.
MOLE = {
    'caco2_wang':                       (0.329, 0.008),
    'hia_hou':                          (0.984, 0.005),
    'pgp_broccatelli':                  (0.930, 0.005),
    'bioavailability_ma':               (0.640, 0.046),
    'lipophilicity_astrazeneca':        (0.406, 0.009),
    'solubility_aqsoldb':               (0.776, 0.019),
    'bbb_martins':                      (0.903, 0.003),
    'ppbr_az':                          (7.229, 0.168),
    'vdss_lombardo':                    (0.644, 0.013),
    'cyp2d6_veith':                     (0.679, 0.006),
    'cyp3a4_veith':                     (0.876, 0.002),
    'cyp2c9_veith':                     (0.782, 0.001),
    'cyp2d6_substrate_carbonmangels':   (0.692, 0.017),
    'cyp3a4_substrate_carbonmangels':   (0.692, 0.019),
    'cyp2c9_substrate_carbonmangels':   (0.409, 0.014),
    'half_life_obach':                  (0.578, 0.032),
    'clearance_microsome_az':           (0.632, 0.008),
    'clearance_hepatocyte_az':          (0.456, 0.027),
    'herg':                             (0.835, 0.018),
    'ames':                             (0.834, 0.015),
    'dili':                             (0.852, 0.022),
    'ld50_zhu':                         (0.602, 0.016),
}

# KPGT per-task ADMET results from
# "A knowledge-guided pre-training framework for improving molecular
#  representation learning" (Nature Communications 2023; doi:10.1038/s41467-023-43214-1),
# Supplementary Table 8 (page 32). Mean ± std over 5 independent runs.
KPGT = {
    'caco2_wang':                       (0.284, 0.009),
    'hia_hou':                          (0.982, 0.004),
    'pgp_broccatelli':                  (0.938, 0.004),
    'bioavailability_ma':               (0.750, 0.022),
    'lipophilicity_astrazeneca':        (0.446, 0.016),
    'solubility_aqsoldb':               (0.714, 0.011),
    'bbb_martins':                      (0.908, 0.005),
    'ppbr_az':                          (7.684, 0.250),
    'vdss_lombardo':                    (0.633, 0.016),
    'cyp2d6_veith':                     (0.724, 0.008),
    'cyp3a4_veith':                     (0.894, 0.004),
    'cyp2c9_veith':                     (0.797, 0.006),
    'cyp2c9_substrate_carbonmangels':   (0.450, 0.044),
    'cyp2d6_substrate_carbonmangels':   (0.737, 0.016),
    'cyp3a4_substrate_carbonmangels':   (0.730, 0.023),
    'half_life_obach':                  (0.531, 0.030),
    'clearance_hepatocyte_az':          (0.424, 0.019),
    'clearance_microsome_az':           (0.637, 0.010),
    'ld50_zhu':                         (0.545, 0.010),
    'herg':                             (0.847, 0.024),
    'ames':                             (0.868, 0.003),
    'dili':                             (0.929, 0.013),
}

# QIP per-task ADMET results from
# Kim et al., "Quantum-Informed Molecular Representation Learning Enhancing
#  ADMET Property Prediction" (J. Chem. Inf. Model. 2024;
#  doi:10.1021/acs.jcim.4c00772), Table 1 ("Ours (HAD)" column).
QIP = {
    'caco2_wang':                       (0.268, 0.012),
    'hia_hou':                          (0.995, 0.004),
    'pgp_broccatelli':                  (0.927, 0.005),
    'bioavailability_ma':               (0.725, 0.017),
    'lipophilicity_astrazeneca':        (0.442, 0.005),
    'solubility_aqsoldb':               (0.707, 0.007),
    'bbb_martins':                      (0.902, 0.009),
    'ppbr_az':                          (7.364, 0.069),
    'vdss_lombardo':                    (0.608, 0.038),
    'cyp2c9_veith':                     (0.787, 0.011),
    'cyp2d6_veith':                     (0.657, 0.015),
    'cyp3a4_veith':                     (0.871, 0.006),
    'cyp2c9_substrate_carbonmangels':   (0.519, 0.036),
    'cyp2d6_substrate_carbonmangels':   (0.665, 0.028),
    'cyp3a4_substrate_carbonmangels':   (0.624, 0.037),
    'half_life_obach':                  (0.529, 0.047),
    'clearance_hepatocyte_az':          (0.508, 0.035),
    'clearance_microsome_az':           (0.658, 0.005),
    'ld50_zhu':                         (0.562, 0.012),
    'herg':                             (0.813, 0.006),
    'ames':                             (0.857, 0.010),
    'dili':                             (0.885, 0.023),
}

# MiniMol (GINE) per-task ADMET results from the official repo README.
# Source: https://github.com/graphcore-research/minimol
MINIMOL_GIT = {
    'caco2_wang':                       (0.350, 0.018),
    'bioavailability_ma':               (0.689, 0.020),
    'lipophilicity_astrazeneca':        (0.456, 0.008),
    'solubility_aqsoldb':               (0.741, 0.013),
    'hia_hou':                          (0.993, 0.005),
    'pgp_broccatelli':                  (0.942, 0.002),
    'bbb_martins':                      (0.924, 0.003),
    'ppbr_az':                          (7.696, 0.125),
    'vdss_lombardo':                    (0.535, 0.027),
    'cyp2c9_veith':                     (0.823, 0.006),
    'cyp2d6_veith':                     (0.719, 0.004),
    'cyp3a4_veith':                     (0.877, 0.001),
    'cyp2c9_substrate_carbonmangels':   (0.474, 0.025),
    'cyp2d6_substrate_carbonmangels':   (0.695, 0.032),
    'cyp3a4_substrate_carbonmangels':   (0.663, 0.008),
    'half_life_obach':                  (0.495, 0.042),
    'clearance_hepatocyte_az':          (0.446, 0.029),
    'clearance_microsome_az':           (0.628, 0.005),
    'ld50_zhu':                         (0.585, 0.005),
    'herg':                             (0.846, 0.016),
    'ames':                             (0.849, 0.004),
    'dili':                             (0.956, 0.006),
}


PROBE_ENSEMBLE_CSV = Path(__file__).parent / '../results/pairmixer_minimol_probe_ensemble.csv'


def _select_probe_rows(sub, selection='latest'):
    """Select one probe row per task.

    Parameters
    ----------
    sub : DataFrame
        Probe-ensemble rows for a fixed (model, pretrain[, ckpt subset]).
    selection : {'latest', 'best'}
        ``latest`` keeps the most recent timestamp per task.
        ``best`` keeps the best ``metric_mean`` per task using ``TASK_METRICS`` to
        decide whether lower or higher is better; ties break by latest timestamp.
    """
    sub = sub.copy()
    sub['timestamp'] = pd.to_datetime(sub['timestamp'], errors='coerce')

    if selection == 'latest':
        return sub.sort_values('timestamp').drop_duplicates(subset=['task'], keep='last')
    if selection != 'best':
        raise ValueError(f'Unknown probe selection mode: {selection}')

    sub['_metric_mean'] = pd.to_numeric(sub['metric_mean'], errors='coerce')
    rows = []
    for task, grp in sub.groupby('task', sort=False):
        valid = grp[grp['_metric_mean'].notna()].copy()
        if valid.empty:
            chosen = grp.sort_values('timestamp').tail(1)
        else:
            direction = TASK_METRICS.get(task, (None, 'higher'))[1]
            best_val = valid['_metric_mean'].min() if direction == 'lower' else valid['_metric_mean'].max()
            chosen = valid[valid['_metric_mean'] == best_val].sort_values('timestamp').tail(1)
        rows.append(chosen)

    if not rows:
        return sub.iloc[0:0]
    return pd.concat(rows, ignore_index=False)


def load_probe_ensemble_dict(
    pretrain_label,
    model='pairmixer_12M',
    csv_path=None,
    selection='latest',
    ckpt_tag_contains=None,
    ckpt_tags=None,
):
    """Load the MiniMol-style probe-ensemble results and return ``{task: (mean, std)}``.

    Reads ``results/pairmixer_minimol_probe_ensemble.csv`` and selects rows matching
    ``(model, pretrain_label)`` plus optional ckpt-tag filters. Replicate rows for the
    same task are reduced according to ``selection``:
      - ``latest``: latest timestamp
      - ``best``: best ``metric_mean`` per task, with latest timestamp as tie-breaker

    Returns an empty dict if the file or row group is missing.
    """
    if csv_path is None:
        csv_path = PROBE_ENSEMBLE_CSV
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    sub = df[(df['model'] == model) & (df['pretrain'] == pretrain_label)].copy()
    if sub.empty:
        return {}
    if ckpt_tag_contains is not None:
        sub = sub[sub['ckpt_tag'].astype(str).str.contains(ckpt_tag_contains, na=False)]
    if ckpt_tags is not None:
        sub = sub[sub['ckpt_tag'].astype(str).isin(list(ckpt_tags))]
    if sub.empty:
        return {}
    sub = _select_probe_rows(sub, selection=selection)
    return {row['task']: (float(row['metric_mean']), float(row['metric_std']))
            for _, row in sub.iterrows()}


def get_sota(task):
    """Return (mean, std) for a task, regardless of the 3-tuple storage format."""
    entry = TDC_SOTA.get(task)
    if entry is None:
        return None
    return (entry[0], entry[1])


def load_tdc_top10():
    """Load TDC top-10 scores and compute derived stats.

    Returns (TDC_TOP10_ALL, TDC_TOP10_AVG, TDC_TOP10_MEDIAN).
    """
    with open(TDC_TOP10_JSON) as f:
        top10_all = json.load(f)
    top10_avg = {}
    top10_median = {}
    for task, scores in top10_all.items():
        top10_avg[task] = (statistics.mean(scores), statistics.pstdev(scores))
        top10_median[task] = statistics.median(scores)
    return top10_all, top10_avg, top10_median


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    ('pyg:gps', 1536, 12): 'gpspp_800M',
    ('pyg:gps', 1472,  8): 'gpspp',
    ('pyg:pairformer', 256,  12): 'pairformer_17M',
    ('pyg:pairformer', 384,  16): 'pairformer_52M',
    ('pyg:pairformer', 384,  48): 'pairformer_boltz',
    ('pyg:pairmixer',  96,   8): 'pairmixer_12M',
    ('pyg:pairmixer',  96,  10): 'pairmixer_16M',
    ('pyg:pairmixer', 256,  18): 'pairmixer_10M',
    ('pyg:pairmixer', 256,  20): 'pairmixer_20M',
    ('pyg:pairmixer', 256,  24): 'pairmixer_40M',
    ('pyg:pairmixer', 384,  48): 'pairmixer_boltz',
}

MODEL_FAMILY_PREFIX = {
    'pyg:gps': 'gpspp',
    'pyg:pairformer': 'pairformer',
    'pyg:pairmixer': 'pairmixer',
}

FAMILY_COLORS = {'GPS++': '#466eff', 'Pairformer': '#EB423D', 'PairMixer': '#2ca02c'}
FAMILY_HATCHES = {'GPS++': '', 'Pairformer': '///', 'PairMixer': '\\\\\\'}


def infer_model_name(row):
    """Map (model_type, hidden_dim, gnn_depth) to a human-readable name."""
    m = str(row.get('model', ''))
    dim_raw = row.get('hidden_dim', 0)
    depth_raw = row.get('gnn_depth', 0)
    if pd.isna(dim_raw) or pd.isna(depth_raw):
        return 'unknown'
    key = (m, int(dim_raw), int(depth_raw))
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    prefix = MODEL_FAMILY_PREFIX.get(m, m)
    return f'{prefix}_{int(dim_raw)}d{int(depth_raw)}'


def infer_model_family(name):
    """Extract the architecture family from a model name."""
    if name.startswith('gpspp'):      return 'GPS++'
    if name.startswith('pairformer'): return 'Pairformer'
    if name.startswith('pairmixer'):  return 'PairMixer'
    return name


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-training dataset parsing
# ═══════════════════════════════════════════════════════════════════════════════

_PRETRAIN_PATTERNS = [
    # Three-component patterns MUST come before two-component ones
    ('toymix_rxrx3_dti',   ['toymix_rxrx3_dti', 'toymix-rxrx3-dti', 'toymixrxrx3dti']),
    ('largemix_rxrx3_dti', ['largemix_rxrx3_dti', 'largemix-rxrx3-dti', 'largemixrxrx3dti']),
    # 4dset combined pretrain — checked first because its checkpoint path
    # contains 'toymix_dti_esmc_v2' as a substring and would otherwise be
    # misidentified as the plain DTI v2 ckpt.
    ('4dset_ep99', ['4dset_ep99', '4dset-ep99', 'lpm24_litopenai_bbbc047', 'lpm24-litopenai-bbbc047']),
    # DTI v2 variants (MUST come before toymix_dti / dti patterns)
    ('toymix_dti_esmc_v2', ['toymix_dti_esmc_v2', 'toymix-dti-esmc-v2']),
    ('toymix_dti_v2',      ['toymix_dti_v2', 'toymix-dti-v2']),
    # DTI pActivity variants (MUST come before toymix_dti / dti / bbbc047 patterns)
    ('bbbc047_dti_pactivity', ['bbbc047_dti_pactivity', 'bbbc047-dti-pactivity']),
    ('toymix_dti_pactivity',  ['toymix_dti_pactivity', 'toymix-dti-pactivity']),
    ('dti_pactivity',         ['dti_pactivity', 'dti-pactivity']),
    # Filtered dataset variants (must come before unfiltered)
    ('toymix_dti_10k_filtered', ['toymix_dti_10k_filtered', 'toymix-dti-10k-filtered']),
    ('toymix_dti_filtered',     ['toymix_dti_filtered', 'toymix-dti-filtered']),
    ('dti_10k_filtered',        ['dti_10k_filtered', 'dti-10k-filtered']),
    ('dti_filtered',            ['dti_filtered', 'dti-filtered']),
    # Two-component patterns
    ('largemix_rxrx3', ['largemix_rxrx3', 'largemixrxrx3', 'largemix-rxrx3']),
    ('largemix_dti',   ['largemix_dti', 'largemixdti', 'largemix-dti']),
    ('toymix_rxrx3',   ['toymix_rxrx3', 'toymixrxrx3', 'toymix-rxrx3']),
    ('toymix_dti',     ['toymix_dti', 'toymixdti', 'toymix-dti']),
    ('rxrx3_dti',      ['rxrx3_dti', 'rxrx3-dti', 'rxrx3dti', 'dti_rxrx3', 'dti-rxrx3', 'dtirxrx3']),
    # BBBC047 patterns (must come before toymix)
    ('toymix_bbbc047', ['toymix_bbbc047', 'toymix-bbbc047']),
    # LPM24 patterns (must come before toymix)
    ('toymix_lpm24',   ['toymix_lpm24', 'toymix-lpm24']),
    # Single-component patterns
    ('largemix',       ['largemix', 'large-dataset']),
    ('toymix',         ['toymix', 'small-dataset']),
    ('rxrx3',          ['rxrx3']),
    ('bbbc047',        ['bbbc047']),
    ('dti',            ['/dti/', '/dti-']),
]


def parse_pretrain_dataset(pretrain_model_path):
    """Infer the pre-training dataset from a checkpoint path or label."""
    s = str(pretrain_model_path).lower()
    if s == 'scratch' or s == 'nan':
        return 'scratch'
    for name, patterns in _PRETRAIN_PATTERNS:
        if any(p in s for p in patterns):
            return name
    return 'unknown'


def parse_data_fraction(row):
    """Extract data fraction from W&B tags, or from known checkpoint mapping."""
    tags = str(row.get('wandb_tags', ''))
    match = re.search(r'frac_(\d+\.?\d*)', tags)
    if match:
        return float(match.group(1))
    KNOWN_FRACS = {
        '2026-03-18_14-53-57': 0.5,
    }
    pt = str(row.get('pretrain_dataset', ''))
    for ts, frac in KNOWN_FRACS.items():
        if ts in pt:
            return frac
    return 1.0


def build_method_label(row):
    """Create a human-readable method label for the results table."""
    ft = row.get('is_finetuning', False)
    pt = row.get('pretrain_type', 'scratch')
    frac = row.get('data_fraction', 1.0)
    frac_str = f' {frac:.0%}' if frac < 1.0 else ''
    if not ft and pt == 'scratch':
        return 'Scratch'
    if ft:
        return f'{pt}{frac_str}'
    return pt


# ═══════════════════════════════════════════════════════════════════════════════
# Results loading pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def find_metric_column(df, task, metric_name):
    """Find the test metric column for a given task."""
    candidates = [c for c in df.columns if task in c.lower() and metric_name in c.lower() and 'test' in c.lower()]
    if candidates:
        return candidates[0]
    candidates = [c for c in df.columns if 'loss' in c.lower() and 'test' in c.lower()]
    return candidates[0] if candidates else None


def load_results(csv_path=None, models_to_show=None,
                 exclude_pretrain=None, min_data_fraction=1.0):
    """Load experiment_results.csv and return (df, admet_df) with all derived columns.

    Parameters
    ----------
    csv_path : Path or str, optional
        Override the default RESULTS_CSV path.
    models_to_show : list of str, optional
        Filter to these model names. None = keep all.
    exclude_pretrain : set of str, optional
        Pretrain types to exclude. Defaults to known bad ones.
    min_data_fraction : float
        Minimum data fraction to keep (default 1.0 = full runs only).

    Returns
    -------
    df : DataFrame
        Full results with derived columns (model_name, pretrain_type, method, etc.)
    admet_df : DataFrame
        Filtered to ADMET tasks, deduped to latest run per (task, model_name, method).
    """
    if csv_path is None:
        csv_path = RESULTS_CSV
    if exclude_pretrain is None:
        exclude_pretrain = {'unknown', 'toymix_dti_10k_filtered', 'dti_10k_filtered', 'toymix 50%'}

    df = pd.read_csv(csv_path)

    # Model name inference
    df['model_name'] = df.apply(infer_model_name, axis=1)
    df['model_family'] = df['model_name'].apply(infer_model_family)

    if models_to_show is not None:
        df = df[df['model_name'].isin(models_to_show)].reset_index(drop=True)

    # Primary metric extraction
    def _extract(row):
        task = row.get('task', '')
        if task not in TASK_METRICS:
            return np.nan
        metric_name, _ = TASK_METRICS[task]
        col = find_metric_column(df, task, metric_name)
        if col is None:
            return np.nan
        return row.get(col, np.nan)

    df['primary_metric'] = df.apply(_extract, axis=1)
    df['primary_metric'] = pd.to_numeric(df['primary_metric'], errors='coerce')
    df['metric_direction'] = df['task'].map(lambda t: TASK_METRICS.get(t, (None, None))[1])
    df['metric_name'] = df['task'].map(lambda t: TASK_METRICS.get(t, (None, None))[0])
    df['category'] = df['task'].map(TASK_CATEGORY)

    # Pretrain parsing
    df['pretrain_type'] = df['pretrain_dataset'].apply(parse_pretrain_dataset)
    df['data_fraction'] = df.apply(parse_data_fraction, axis=1)
    df['method'] = df.apply(build_method_label, axis=1)

    # Build admet_df
    admet_df = df[df['task'].isin(TASK_METRICS.keys())].copy()
    admet_df = admet_df[~admet_df['pretrain_type'].isin(exclude_pretrain)]
    admet_df = admet_df[admet_df['data_fraction'] >= min_data_fraction]

    # Dedup to latest run per (task, model_name, method)
    admet_df['timestamp'] = pd.to_datetime(admet_df['timestamp'], errors='coerce')
    admet_df = admet_df.sort_values('timestamp').drop_duplicates(
        subset=['task', 'model_name', 'method'], keep='last'
    ).reset_index(drop=True)

    return df, admet_df


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_scores(admet_df):
    """Normalize primary_metric within each task to [0, 1] where 1 = best.

    Returns admet_df with a new 'normalized_score' column.
    """
    admet_df = admet_df.copy()
    parts = []
    for task, group in admet_df.groupby('task'):
        direction = TASK_METRICS.get(task, (None, 'higher'))[1]
        vals = group['primary_metric'].astype(float)
        if direction == 'lower':
            vals = -vals
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            norm = pd.Series(0.5, index=group.index)
        else:
            norm = (vals - vmin) / (vmax - vmin)
        parts.append(norm)
    admet_df['normalized_score'] = pd.concat(parts).astype(float)
    return admet_df


def count_best_among_methods(task_method_pivot, methods, task_metrics=None):
    """Count how many tasks each method is the best performer."""
    if task_metrics is None:
        task_metrics = TASK_METRICS
    wins = Counter()
    n_tasks = 0
    for task in task_method_pivot.index:
        if task not in task_metrics:
            continue
        direction = task_metrics[task][1]
        row = task_method_pivot.loc[task, [m for m in methods if m in task_method_pivot.columns]].dropna()
        if len(row) == 0:
            continue
        n_tasks += 1
        best = row.idxmin() if direction == 'lower' else row.idxmax()
        wins[best] += 1
    return {m: (wins.get(m, 0), n_tasks) for m in methods if m in task_method_pivot.columns}


# ═══════════════════════════════════════════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════════════════════════════════════════

METHOD_PALETTE = ['#466eff', '#64B478', '#EB423D', '#FF883D', '#9B59B6',
                  '#17becf', '#bcbd22', '#e377c2', '#d62728', '#8c564b']


def build_method_colors(admet_df):
    """Build a method -> color mapping from the data, with Scratch forced to grey."""
    colors = {'Scratch': '#888888'}
    all_methods = sorted(admet_df['method'].unique())
    idx = 0
    for m in all_methods:
        if m not in colors:
            colors[m] = METHOD_PALETTE[idx % len(METHOD_PALETTE)]
            idx += 1
    return colors
