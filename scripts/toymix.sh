cd ../../

CUDA_VISIBLE_DEVICES=$1 graphium-train \
    architecture=toymix \
    tasks=toymix \
    training=toymix \
    model=$2 \
    accelerator=gpu \
    ++datamodule.args.task_specific_args.qm9.df_path=./data/graphium/neurips2023/small-dataset/qm9.csv \
    ++datamodule.args.task_specific_args.qm9.splits_path=./data/graphium/neurips2023/small-dataset/qm9_random_splits.pt \
    ++datamodule.args.task_specific_args.tox21.df_path=./data/graphium/neurips2023/small-dataset/Tox21-7k-12-labels.csv \
    ++datamodule.args.task_specific_args.tox21.splits_path=./data/graphium/neurips2023/small-dataset/Tox21_random_splits.pt \
    ++datamodule.args.task_specific_args.zinc.df_path=./data/graphium/neurips2023/small-dataset/ZINC12k.csv \
    ++datamodule.args.task_specific_args.zinc.splits_path=./data/graphium/neurips2023/small-dataset/ZINC12k_random_splits.pt \
    ++datamodule.args.processed_graph_data_path=./data/graphium/toymix
