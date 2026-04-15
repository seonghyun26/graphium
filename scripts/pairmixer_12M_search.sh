#!/usr/bin/env bash
# PairMixer ~12M architecture search on 5 representative ADMET tasks.
#
# Runs `pairmixer_auto` with CLI overrides for each (D_s, D_z, depth) point.
# Scratch training (no pretrain), seed=0, 100 epochs, GPU 0, no checkpoints.
#
# Phase: 1 (architecture) | 2 (pair_init on best arch) — selected via $PHASE.
# Variant / task lists are overridable via env vars.
#
# Usage:
#   bash scripts/pairmixer_12M_search.sh              # phase 1
#   BEST_CFG=C bash scripts/pairmixer_12M_search.sh 2 # phase 2 on config C
#
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/common.sh"

PHASE=${1:-${PHASE:-1}}
DEVICE=${DEVICE:-0}
SEED=${SEED:-0}
MAX_EPOCHS=${MAX_EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
MODEL=pairmixer_auto

# ── 5 representative ADMET tasks (one per category) ──────────────────────────
TASKS=(
    caco2_wang               # Absorption  — regression, mae
    bbb_martins              # Distribution — classification, auroc
    cyp2d6_veith             # Metabolism  — classification, auprc
    clearance_microsome_az   # Excretion   — regression, spearman
    ld50_zhu                 # Toxicity    — regression, mae
)

# ── Arch overrides per variant ───────────────────────────────────────────────
arch_flags_for () {
    local name=$1
    local dsf dsn dz depth
    case "${name}" in
        B)  dsf=128; dz=160; depth=16 ;;   # 12.0 M — depth-heavy
        C)  dsf=128; dz=192; depth=12 ;;   # 12.7 M — balanced
        D)  dsf=128; dz=256; depth=7  ;;   # 13.2 M — wide/shallow (over budget)
        E)  dsf=192; dz=192; depth=10 ;;   # 11.3 M — wide-single
        F1) dsf=96;  dz=256; depth=6  ;;   # 11.52 M — narrow-s, wide-z, shallow
        F2) dsf=128; dz=256; depth=6  ;;   # 11.62 M — D minus one layer (≤12M)
        F3) dsf=96;  dz=224; depth=8  ;;   # 11.62 M — intermediate width × deeper
        F4) dsf=192; dz=192; depth=11 ;;   # ~11.8 M — E + one layer
        *)  echo "Unknown config ${name}" >&2; return 1 ;;
    esac
    echo "++architecture.pre_nn.out_dim=${dsf}"
    echo "++architecture.pre_nn.hidden_dims=${dsf}"
    echo "++architecture.gnn.in_dim=${dsf}"
    echo "++architecture.gnn.out_dim=${dsf}"
    echo "++architecture.gnn.hidden_dims=${dsf}"
    echo "++architecture.gnn.depth=${depth}"
    echo "++architecture.gnn.layer_kwargs.pair_dim=${dz}"
    echo "++architecture.graph_output_nn.graph.pair_dim=${dz}"
}

# ── Pair-init overrides per variant (phase 2) ────────────────────────────────
pair_init_flags_for () {
    local name=$1
    case "${name}" in
        opm)
            # Default: OPM-only. Nothing to add.
            ;;
        struct)
            # OPM + graph_distance + adjacency
            echo "++architecture.gnn.layer_kwargs.pair_init.mode=additive"
            echo "++architecture.gnn.layer_kwargs.pair_init.use_opm=true"
            echo "++architecture.gnn.layer_kwargs.pair_init.opm_init=lecun"
            echo "++architecture.gnn.layer_kwargs.pair_init.use_pair_positional=true"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.sources=[graph_distance,adjacency]"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.graph_distance.max_dist=8"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.graph_distance.embedding_dim=32"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.adjacency.embedding_dim=8"
            ;;
        struct_ef)
            # struct + raw edge_feat
            echo "++architecture.gnn.layer_kwargs.pair_init.mode=additive"
            echo "++architecture.gnn.layer_kwargs.pair_init.use_opm=true"
            echo "++architecture.gnn.layer_kwargs.pair_init.opm_init=lecun"
            echo "++architecture.gnn.layer_kwargs.pair_init.use_pair_positional=true"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.sources=[graph_distance,adjacency,edge_feat]"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.graph_distance.max_dist=8"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.graph_distance.embedding_dim=32"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.adjacency.embedding_dim=8"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.edge_feat.embedding_dim=16"
            ;;
        struct_path)
            # struct + path_edge
            echo "++architecture.gnn.layer_kwargs.pair_init.mode=additive"
            echo "++architecture.gnn.layer_kwargs.pair_init.use_opm=true"
            echo "++architecture.gnn.layer_kwargs.pair_init.opm_init=lecun"
            echo "++architecture.gnn.layer_kwargs.pair_init.use_pair_positional=true"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.sources=[graph_distance,adjacency,path_edge]"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.graph_distance.max_dist=8"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.graph_distance.embedding_dim=32"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.adjacency.embedding_dim=8"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.path_edge.embedding_dim=16"
            echo "++architecture.gnn.layer_kwargs.pair_init.pair_positional.path_edge.max_path_length=5"
            ;;
        *)  echo "Unknown pair_init variant ${name}" >&2; return 1 ;;
    esac
}

# ── Phase selection ──────────────────────────────────────────────────────────
if [[ "${PHASE}" == "1" ]]; then
    if [[ -n "${CONFIGS_OVERRIDE:-}" ]]; then
        read -ra CONFIGS <<< "${CONFIGS_OVERRIDE}"
    else
        CONFIGS=(B C D E)
    fi
    PAIR_INIT_NAME=opm
    PHASE_TAG=arch
elif [[ "${PHASE}" == "2" ]]; then
    BEST_CFG=${BEST_CFG:?  "PHASE=2 requires BEST_CFG=<B|C|D|E>"}
    CONFIGS=(${BEST_CFG})
    if [[ -n "${PAIR_INIT_OVERRIDE:-}" ]]; then
        read -ra PAIR_INIT_VARIANTS <<< "${PAIR_INIT_OVERRIDE}"
    else
        PAIR_INIT_VARIANTS=(opm struct struct_ef struct_path)
    fi
    PHASE_TAG=pairinit
elif [[ "${PHASE}" == "3" ]]; then
    # Refined architecture + enhanced featurization (all strict ≤12M).
    if [[ -n "${CONFIGS_OVERRIDE:-}" ]]; then
        read -ra CONFIGS <<< "${CONFIGS_OVERRIDE}"
    else
        CONFIGS=(F1 F2 F3 F4)
    fi
    PAIR_INIT_NAME=opm
    PHASE_TAG=arch2
    USE_FEATURES2=1
elif [[ "${PHASE}" == "4" ]]; then
    # Cross-phase: best arch + best pair_init + expanded features.
    BEST_CFG=${BEST_CFG:?  "PHASE=4 requires BEST_CFG=<F1|F2|F3|F4|...>"}
    CONFIGS=(${BEST_CFG})
    if [[ -n "${PAIR_INIT_OVERRIDE:-}" ]]; then
        read -ra PAIR_INIT_VARIANTS <<< "${PAIR_INIT_OVERRIDE}"
    else
        PAIR_INIT_VARIANTS=(struct struct_ef struct_path)
    fi
    PHASE_TAG=combo
    USE_FEATURES2=1
else
    echo "Unknown PHASE=${PHASE}; use 1 or 2" >&2
    exit 1
fi

# ── Expanded featurization (shared by PHASE=3 and PHASE=4) ──────────────────
if [[ "${USE_FEATURES2:-0}" == "1" ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS:-} \
        ++datamodule.args.featurization.atom_property_list_onehot=[atomic-number,group,period,total-valence,hybridization,chirality] \
        ++datamodule.args.featurization.atom_property_list_float=[degree,formal-charge,radical-electron,aromatic,in-ring,mass,electronegativity,vdw-radius,num-ring] \
        ++datamodule.args.featurization.edge_property_list=[bond-type-onehot,stereo,in-ring,conjugated]"
    # Unique datacache per (cfg, pi) group to avoid featurization races.
    CACHE_KEY="${CONFIGS_OVERRIDE:-${BEST_CFG:-x}}"
    CACHE_KEY="${CACHE_KEY// /_}"
    CACHE_KEY="${CACHE_KEY}_${PAIR_INIT_OVERRIDE// /_}"
    export DATACACHE_SUFFIX="-feat2-${CACHE_KEY}"
fi

# ── Main loop ────────────────────────────────────────────────────────────────
for CFG in "${CONFIGS[@]}"; do
    VARIANTS=(${PAIR_INIT_NAME:-${PAIR_INIT_VARIANTS[@]}})
    for PI in "${VARIANTS[@]}"; do
        RUN_NAME="pm_auto_${CFG}_${PI}"
        TAGS="['pairmixer_auto','scratch','admet','${PHASE_TAG}_search','cfg_${CFG}','pi_${PI}']"
        ARCH_FLAGS=$(arch_flags_for "${CFG}")
        PI_FLAGS=$(pair_init_flags_for "${PI}")

        for task in "${TASKS[@]}"; do
            echo ""
            echo "╔══════════════════════════════════════════════════════════════"
            echo "║ PHASE=${PHASE}  cfg=${CFG}  pair_init=${PI}  task=${task}"
            echo "╚══════════════════════════════════════════════════════════════"

            CUDA_VISIBLE_DEVICES=${DEVICE} graphium-train \
                model=${MODEL} \
                accelerator=gpu \
                tasks=admet \
                ${DATACACHE_SUFFIX:+++constants.datacache_path=../datacache/neurips2023-small${DATACACHE_SUFFIX}/} \
                $(wandb_flags "${TAGS}") \
                ++constants.raise_train_error=False \
                ++constants.name=${RUN_NAME}_${task} \
                ++constants.max_epochs=${MAX_EPOCHS} \
                ++constants.task=${task} \
                ++datamodule.args.tdc_benchmark_names=${task} \
                ++datamodule.args.batch_size_training=${BATCH_SIZE} \
                ++datamodule.args.batch_size_inference=${BATCH_SIZE} \
                ${ARCH_FLAGS} \
                ${PI_FLAGS} \
                ++architecture.task_heads.${task}.hidden_dims=256 \
                ++architecture.task_heads.${task}.depth=4 \
                ++trainer.model_checkpoint.save_last=False \
                ++trainer.model_checkpoint.save_top_k=0 \
                ++trainer.model_checkpoint.dirpath=/tmp/pm_search_ckpt/ \
                ${EXTRA_FLAGS:-} \
            || echo "WARN: cfg=${CFG} pi=${PI} task=${task} failed, continuing..."
        done
    done
done

echo ""
echo "=== PairMixer ${PHASE_TAG} search (phase ${PHASE}) complete ==="
