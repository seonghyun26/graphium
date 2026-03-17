#!/usr/bin/env bash
# Shared configuration for all experiment scripts.
# Source this file: source "$(dirname "$0")/common.sh"

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Defaults ─────────────────────────────────────────────────────────────────
DEVICE=${DEVICE:-0}
SEED=${SEED:-0}

# ── W&B ──────────────────────────────────────────────────────────────────────
WANDB_ENTITY=${WANDB_ENTITY:-eddy26}
WANDB_PROJECT=${WANDB_PROJECT:-graphium}

# ── ADMET task list (22 TDC benchmark tasks) ─────────────────────────────────
ADMET_TASKS=(
    # Absorption
    caco2_wang hia_hou pgp_broccatelli bioavailability_ma
    lipophilicity_astrazeneca solubility_aqsoldb bbb_martins ppbr_az vdss_lombardo
    # Metabolism
    cyp2d6_veith cyp3a4_veith cyp2c9_veith
    cyp2c9_substrate_carbonmangels cyp2d6_substrate_carbonmangels cyp3a4_substrate_carbonmangels
    # Excretion
    half_life_obach clearance_hepatocyte_az clearance_microsome_az
    # Toxicity
    ld50_zhu herg ames dili
)

# ── Helper: common W&B flags ─────────────────────────────────────────────────
wandb_flags() {
    local tags="$1"
    echo "++constants.seed=${SEED}"
    echo "++constants.wandb.entity=${WANDB_ENTITY}"
    echo "++constants.wandb.save_dir=null"
    echo "++constants.wandb.project=${WANDB_PROJECT}"
    echo "++constants.wandb.tags=${tags}"
}

# ── Helper: set architecture dims for GCN-family models ──────────────────────
gcn_dim_flags() {
    local dim=$1
    echo "++architecture.pre_nn.out_dim=${dim}"
    echo "++architecture.gnn.in_dim=${dim}"
    echo "++architecture.gnn.out_dim=${dim}"
    echo "++architecture.gnn.hidden_dims=${dim}"
    echo "++architecture.graph_output_nn.graph.hidden_dims=${dim}"
    echo "++architecture.graph_output_nn.graph.out_dim=${dim}"
}

# ── Helper: set architecture dims for MPNN-family models ─────────────────────
mpnn_dim_flags() {
    local dim=$1
    echo "++architecture.pre_nn.hidden_dims=${dim}"
    echo "++architecture.pre_nn.out_dim=${dim}"
    echo "++architecture.pre_nn_edges.out_dim=${dim}"
    echo "++architecture.pre_nn_edges.hidden_dims=${dim}"
    echo "++architecture.gnn.in_dim=${dim}"
    echo "++architecture.gnn.out_dim=${dim}"
    echo "++constants.gnn_edge_dim=${dim}"
    echo "++architecture.gnn.hidden_dims=${dim}"
    echo "++architecture.graph_output_nn.graph.hidden_dims=${dim}"
    echo "++architecture.graph_output_nn.graph.out_dim=${dim}"
}

# ── Helper: set architecture dims for GPS++ models ───────────────────────────
gpspp_dim_flags() {
    local dim=$1
    echo "++architecture.pre_nn.out_dim=${dim}"
    echo "++architecture.pre_nn.hidden_dims=${dim}"
    echo "++architecture.gnn.layer_kwargs.mpnn_kwargs.in_dim=${dim}"
    echo "++architecture.gnn.layer_kwargs.mpnn_kwargs.out_dim=${dim}"
    echo "++architecture.gnn.in_dim=${dim}"
    echo "++architecture.gnn.hidden_dims=${dim}"
    echo "++architecture.gnn.out_dim=${dim}"
}
