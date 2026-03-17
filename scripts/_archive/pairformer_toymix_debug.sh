#!/bin/bash
# Quick sanity-check: pairformer on ToyMix
# Runs 3 epochs, 10 train batches, tiny model – should finish in ~1-2 min.
# Usage: bash scripts/pairformer_toymix_debug.sh [GPU_ID]
cd "$(dirname "$0")/.."

DEVICE=${1:-'0'}

CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=pairformer \
    training=toymix \
    tasks=toymix \
    architecture=toymix \
    \
    ++constants.name=pairformer_debug \
    ++constants.max_epochs=3 \
    ++constants.raise_train_error=true \
    \
    ++datamodule.args.batch_size_training=16 \
    ++datamodule.args.task_specific_args.qm9.df_path=./data/graphium/neurips2023/small-dataset/qm9.csv \
    ++datamodule.args.task_specific_args.qm9.splits_path=./data/graphium/neurips2023/small-dataset/qm9_random_splits.pt \
    ++datamodule.args.task_specific_args.tox21.df_path=./data/graphium/neurips2023/small-dataset/Tox21-7k-12-labels.csv \
    ++datamodule.args.task_specific_args.tox21.splits_path=./data/graphium/neurips2023/small-dataset/Tox21_random_splits.pt \
    ++datamodule.args.task_specific_args.zinc.df_path=./data/graphium/neurips2023/small-dataset/ZINC12k.csv \
    ++datamodule.args.task_specific_args.zinc.splits_path=./data/graphium/neurips2023/small-dataset/ZINC12k_random_splits.pt \
    \
    ++architecture.gnn.depth=2 \
    ++architecture.gnn.layer_kwargs.pair_dim=16 \
    ++architecture.gnn.layer_kwargs.num_heads=4 \
    ++architecture.gnn.layer_kwargs.pairwise_num_heads=2 \
    ++architecture.gnn.layer_kwargs.pairwise_head_width=16 \
    ++architecture.gnn.layer_kwargs.opm_hidden=8 \
    ++architecture.gnn.layer_kwargs.use_checkpoint=false \
    \
    ++trainer.trainer.limit_train_batches=10 \
    ++trainer.trainer.limit_val_batches=3 \
    ++trainer.trainer.check_val_every_n_epoch=1
