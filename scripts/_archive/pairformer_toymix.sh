cd ../

DEVICE=${1:-'0'}
# Pairformer pre-training on ToyMix (lightweight dataset for quick iteration)

CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=pairformer \
    accelerator=gpu \
    tasks=toymix \
    training=toymix \
    architecture=toymix \
    ++constants.seed=42 \
    ++datamodule.args.batch_size_training=32 \
    ++datamodule.args.task_specific_args.qm9.df_path=./data/graphium/neurips2023/small-dataset/qm9.csv \
    ++datamodule.args.task_specific_args.qm9.splits_path=./data/graphium/neurips2023/small-dataset/qm9_random_splits.pt \
    ++datamodule.args.task_specific_args.tox21.df_path=./data/graphium/neurips2023/small-dataset/Tox21-7k-12-labels.csv \
    ++datamodule.args.task_specific_args.tox21.splits_path=./data/graphium/neurips2023/small-dataset/Tox21_random_splits.pt \
    ++datamodule.args.task_specific_args.zinc.df_path=./data/graphium/neurips2023/small-dataset/ZINC12k.csv \
    ++datamodule.args.task_specific_args.zinc.splits_path=./data/graphium/neurips2023/small-dataset/ZINC12k_random_splits.pt 
